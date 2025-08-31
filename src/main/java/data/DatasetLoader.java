package data;

import java.io.IOException;
import java.io.BufferedReader;
import java.nio.file.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * טוען קבצי מניות (CSV) מתיקייה, מבצע ניקוי בסיסי, מחשב פיצ'רים ותיוג (5 קטגוריות),
 * ומפצל אימון/בדיקה לפי סדר כרונולוגי בכל מניה בנפרד (ללא דליפת עתיד).
 *
 * צפי עמודות מינימלי: date, open_price, high_price, low_price, close_price, target_close
 * אופציונלי: volume
 */
public class DatasetLoader {

    public static class Dataset {
        public final double[][] Xtrain, Xtest;
        public final int[] ytrain, ytest;
        public final String[] datesTest, tickersTest; // לנוחות בדיווח תוצאות
        public Dataset(double[][] Xtr, int[] ytr, double[][] Xte, int[] yte,
                       String[] datesTest, String[] tickersTest) {
            this.Xtrain = Xtr; this.ytrain = ytr;
            this.Xtest = Xte;  this.ytest = yte;
            this.datesTest = datesTest; this.tickersTest = tickersTest;
        }
    }

    private static final DateTimeFormatter DATE_FMT = DateTimeFormatter.ISO_LOCAL_DATE;

    /** טוען את כל קבצי ה-CSV בתיקייה (למשל: "src/main/resources/data"), עם ε=0.5% וסף "גדול" = 2%. */
    public Dataset loadFolder(String folder,
                              double epsilonPct,
                              double bigMovePct,
                              boolean useLogVolume,
                              boolean addDerived
    ) throws IOException {
        Path root = Paths.get(folder);
        if (!Files.isDirectory(root)) throw new IllegalArgumentException("Folder not found: " + folder);

        List<double[]> XtrAll = new ArrayList<>();
        List<Integer> ytrAll = new ArrayList<>();
        List<double[]> XteAll = new ArrayList<>();
        List<Integer> yteAll = new ArrayList<>();
        List<String> datesTeAll = new ArrayList<>();
        List<String> tickersTeAll = new ArrayList<>();

        try (var stream = Files.list(root)) {
            for (Path p : (Iterable<Path>) stream::iterator) {
                if (Files.isRegularFile(p) && p.getFileName().toString().toLowerCase().endsWith(".csv")) {
                    String ticker = fileBaseName(p.getFileName().toString());
                    PerTickerData td = loadSingleCsv(p, ticker, epsilonPct, bigMovePct, useLogVolume, addDerived);

                    // ---- תקנון לפי טיקר (z-score) על TRAIN בלבד ----
                    int n = td.rows.size();
                    if (n < 10) continue;
                    int split = Math.max(1, (int)Math.floor(n * 0.7)); // 70% אימון, 30% בדיקה
                    int d = td.rows.get(0).x.length;

                    // ממוצעים על Train של אותו טיקר
                    double[] mean = new double[d];
                    for (int i = 0; i < split; i++) {
                        double[] x = td.rows.get(i).x;
                        for (int j = 0; j < d; j++) mean[j] += x[j];
                    }
                    for (int j = 0; j < d; j++) mean[j] /= split;

                    // סטיות תקן על Train של אותו טיקר
                    double[] std = new double[d];
                    for (int i = 0; i < split; i++) {
                        double[] x = td.rows.get(i).x;
                        for (int j = 0; j < d; j++) {
                            double v = x[j] - mean[j];
                            std[j] += v * v;
                        }
                    }
                    for (int j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / Math.max(1, split - 1) + 1e-12);

                    // החלת התקנון על כל השורות של אותו טיקר (Train+Test)
                    for (int i = 0; i < n; i++) {
                        double[] x = td.rows.get(i).x;
                        for (int j = 0; j < d; j++) x[j] = (x[j] - mean[j]) / std[j];
                    }
                    // ---- סוף התקנון ----

                    // העתקה לסט המאוחד
                    for (int i = 0; i < split; i++) {
                        Row r = td.rows.get(i);
                        XtrAll.add(r.x);
                        ytrAll.add(r.y);
                    }
                    for (int i = split; i < n; i++) {
                        Row r = td.rows.get(i);
                        XteAll.add(r.x);
                        yteAll.add(r.y);
                        datesTeAll.add(r.date);
                        tickersTeAll.add(r.ticker);
                    }
                }
            }
        }

        if (XtrAll.isEmpty() || XteAll.isEmpty())
            throw new IllegalStateException("לא נמצאו נתוני אימון/בדיקה. בדקי את התיקייה והקבצים.");

        return new Dataset(
                list2mat(XtrAll), list2int(ytrAll),
                list2mat(XteAll), list2int(yteAll),
                datesTeAll.toArray(new String[0]),
                tickersTeAll.toArray(new String[0])
        );
    }


    // ===== קריאה ועיבוד של קובץ מניה בודד =====

    private static class Row {
        final String date, ticker;
        final double[] x;
        final int y;
        Row(String date, String ticker, double[] x, int y) {
            this.date = date; this.ticker = ticker; this.x = x; this.y = y;
        }
    }

    private static class PerTickerData {
        final List<Row> rows = new ArrayList<>();
    }

    private PerTickerData loadSingleCsv(Path csvPath, String ticker,
                                        double epsilonPct, double bigMovePct,
                                        boolean useLogVolume, boolean addDerived
    ) throws IOException {
        List<String> lines = Files.readAllLines(csvPath);
        if (lines.isEmpty()) throw new IllegalArgumentException("Empty CSV: " + csvPath);

        String[] header = splitCsvLine(lines.get(0));
        Map<String, Integer> idx = indexHeader(header);

        // חובה: date, open/high/low/close, target_close
        int iDate = need(idx, "date");
        int iOpen = need(idx, "open_price");
        int iHigh = need(idx, "high_price");
        int iLow  = need(idx, "low_price");
        int iClose= need(idx, "close_price");
        int iTgt  = need(idx, "target_close");
        Integer iVol = idx.get("volume"); // אופציונלי

        // נאסוף רשומות תקינות
        List<Row> all = new ArrayList<>();
        for (int li = 1; li < lines.size(); li++) {
            String line = lines.get(li).trim();
            if (line.isEmpty()) continue;
            String[] a = splitCsvLine(line);
            if (a.length < header.length) continue;

            String dateStr = a[iDate].trim();
            // סינון: תאריך תקין
            LocalDate date;
            try { date = LocalDate.parse(dateStr, DATE_FMT); }
            catch (Exception e) { continue; }

            Double open = parseD(a[iOpen]);
            Double high = parseD(a[iHigh]);
            Double low  = parseD(a[iLow]);
            Double close= parseD(a[iClose]);
            Double tgt  = parseD(a[iTgt]);
            if (open == null || high == null || low == null || close == null || tgt == null) continue;
            if (open <= 0 || high <= 0 || low <= 0 || close <= 0) continue;

            // פיצ'רים בסיסיים
            List<Double> feats = new ArrayList<>();
            double openRel = (open - close) / close;
            double highRel = (high - close) / close;
            double lowRel  = (low  - close) / close;
            feats.add(openRel);
            feats.add(highRel);
            feats.add(lowRel);

            // volume אופציונלי
            if (useLogVolume && iVol != null) {
                Double vol = parseD(a[iVol]);
                if (vol != null && vol > 0) feats.add(Math.log1p(vol));
            }

            // נגזרים פשוטים (לא חובה)
            if (addDerived) {
                double rangePct = (high - low) / close;                 // טווח יומי ביחס לסגירה
                double intraRet = (close - open) / open;                 // תשואה תוך-יומית
                feats.add(rangePct);
                feats.add(intraRet);
            }

            // תיוג 5 קטגוריות לפי שינוי למחר
            double retPct = (tgt - close) / close * 100.0;
            int y = mapToFiveClasses(retPct, epsilonPct, bigMovePct);

            // בונים שורה
            double[] x = new double[feats.size()];
            for (int i = 0; i < x.length; i++) x[i] = feats.get(i);
            all.add(new Row(dateStr, ticker, x, y));
        }

        // מיון כרונולוגי
        all.sort(Comparator.comparing(r -> LocalDate.parse(r.date, DATE_FMT)));

        PerTickerData td = new PerTickerData();
        td.rows.addAll(all);
        return td;
    }

    // ===== עוזרים =====

    private static String fileBaseName(String name) {
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }

    private static Map<String, Integer> indexHeader(String[] header) {
        Map<String, Integer> m = new HashMap<>();
        for (int i = 0; i < header.length; i++) {
            m.put(header[i].trim().toLowerCase(), i);
        }
        return m;
    }
    private static int need(Map<String,Integer> idx, String key) {
        Integer i = idx.get(key);
        if (i == null) throw new IllegalArgumentException("Column missing: " + key);
        return i;
    }

    private static String[] splitCsvLine(String line) {
        // פיצול פשוט בפסיקים (מספיק אם אין שדות מצוטטים עם פסיקים פנימיים)
        return line.split(",", -1);
    }

    private static Double parseD(String s) {
        try {
            if (s == null) return null;
            s = s.trim();
            if (s.isEmpty()) return null;
            return Double.parseDouble(s);
        } catch (Exception e) {
            return null;
        }
    }

    /** מיפוי ל-5 קטגוריות עם ספים סימטריים סביב 0: ε ו-±bigMove. */
    public static int mapToFiveClasses(double retPct, double epsilon, double bigMove) {
        if (retPct <= -bigMove) return 0;                 // ירידה חדה
        if (retPct <= -epsilon) return 1;                 // ירידה קלה
        if (retPct <  +epsilon) return 2;                 // כמעט ללא שינוי
        if (retPct <  +bigMove) return 3;                 // עלייה קלה
        return 4;                                         // עלייה חדה
    }

    private static double[][] list2mat(List<double[]> rows) {
        double[][] M = new double[rows.size()][];
        for (int i = 0; i < rows.size(); i++) M[i] = rows.get(i);
        return M;
    }
    private static int[] list2int(List<Integer> rows) {
        int[] v = new int[rows.size()];
        for (int i = 0; i < rows.size(); i++) v[i] = rows.get(i);
        return v;
    }

    /** טוען קובץ iris.csv: 4 פיצ'רים מספריים + עמודת תווית.
     *  תומך בכותרת, מירכאות, ומפריד ',' או ';' (אוטו-דיטקט).
     */
    public Dataset loadIris(String csvPath, double trainRatio, long seed) throws IOException {
        List<double[]> feats = new ArrayList<>();
        List<Integer> labs = new ArrayList<>();

        try (BufferedReader br = Files.newBufferedReader(Path.of(csvPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                // אוטו-דיטקט למפריד
                String delim = line.indexOf(';') >= 0 ? ";" : ",";

                // פיצול + הסרת מירכאות מכל תא
                String[] raw = line.split(delim, -1);
                String[] p = new String[raw.length];
                for (int i = 0; i < raw.length; i++) {
                    p[i] = raw[i].trim();
                    if (p[i].length() >= 2 && p[i].startsWith("\"") && p[i].endsWith("\"")) {
                        p[i] = p[i].substring(1, p[i].length() - 1);
                    }
                }

                // דלג על כותרת: אם התא הראשון לא מספרי (למשל "sepal.length")
                if (p.length >= 5) {
                    boolean firstNumeric;
                    try { Double.parseDouble(p[0]); firstNumeric = true; }
                    catch (Exception e) { firstNumeric = false; }
                    if (!firstNumeric) continue; // כותרת או שורה לא תקינה
                } else {
                    continue;
                }

                // קריאת 4 פיצ'רים
                double[] x = new double[4];
                try {
                    for (int i = 0; i < 4; i++) x[i] = Double.parseDouble(p[i]);
                } catch (NumberFormatException nfe) {
                    // אם יש שורה מקולקלת – נדלג עליה
                    continue;
                }

                // מיפוי תווית למחלקה 0/1/2
                String label = p[4].trim();
                label = label.replace("\"", "");
                label = label.toLowerCase(Locale.ROOT);
                if (label.startsWith("iris-")) label = label.substring("iris-".length()); // "iris-setosa" -> "setosa"

                int y = switch (label) {
                    case "setosa" -> 0;
                    case "versicolor" -> 1;
                    case "virginica" -> 2;
                    default -> {
                        // יש קבצים שבהם העמודה נקראת "species" אבל הערך ריק/שונה – נדלג
                        // System.out.println("Skipping unknown iris label: " + p[4]);
                        yield -1;
                    }
                };
                if (y < 0) continue;

                feats.add(x);
                labs.add(y);
            }
        }

        int n = feats.size(), d = 4;
        if (n == 0) throw new IllegalStateException("No Iris rows parsed from: " + csvPath);

        // ערבוב
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Random rnd = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
        }

        int split = Math.max(1, Math.min(n - 1, (int)Math.round(trainRatio * n)));

        double[][] Xtr = new double[split][d];
        double[][] Xte = new double[n - split][d];
        int[] ytr = new int[split];
        int[] yte = new int[n - split];

        for (int i = 0; i < split; i++) { Xtr[i] = feats.get(idx[i]); ytr[i] = labs.get(idx[i]); }
        for (int i = split; i < n; i++) { Xte[i - split] = feats.get(idx[i]); yte[i - split] = labs.get(idx[i]); }

        return new Dataset(Xtr, ytr, Xte, yte, null, null);
    }


}
