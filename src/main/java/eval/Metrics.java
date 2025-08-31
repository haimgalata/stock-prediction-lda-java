package eval;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;

/** מדדים בסיסיים לדיווח תוצאות + שמירת תחזיות ל-CSV. */
public final class Metrics {
    private Metrics() {}

    public static double accuracy(int[] y, int[] yhat) {
        int ok = 0;
        for (int i = 0; i < y.length; i++) if (y[i] == yhat[i]) ok++;
        return (double) ok / y.length;
    }

    public static int[] classCounts(int[] y, int K) {
        int[] c = new int[K];
        for (int v : y) { if (v >= 0 && v < K) c[v]++; }
        return c;
    }

    public static int majorityClass(int[] y, int K) {
        int[] c = classCounts(y, K);
        int arg = 0;
        for (int k = 1; k < K; k++) if (c[k] > c[arg]) arg = k;
        return arg;
    }

    /** דיוק לכל כיתה (Hit-rate) – אחוז פגיעות מתוך הדוגמאות של אותה כיתה. */
    public static double[] perClassHitRate(int[] y, int[] yhat, int K) {
        int[] tot = new int[K];
        int[] hit = new int[K];
        for (int i = 0; i < y.length; i++) {
            int t = y[i];
            if (t >= 0 && t < K) {
                tot[t]++;
                if (yhat[i] == t) hit[t]++;
            }
        }
        double[] r = new double[K];
        for (int k = 0; k < K; k++) r[k] = tot[k] == 0 ? Double.NaN : (double) hit[k] / tot[k];
        return r;
    }

    /** דיוק רק על תוויות מסוימות (למשל big-moves: 0 ו-4). מחזיר [accuracy, n]. */
    public static double[] accuracyOnLabels(int[] y, int[] yhat, int... labels) {
        Set<Integer> set = new HashSet<>();
        for (int v : labels) set.add(v);
        int n = 0, ok = 0;
        for (int i = 0; i < y.length; i++) {
            if (set.contains(y[i])) { n++; if (y[i] == yhat[i]) ok++; }
        }
        double acc = n == 0 ? Double.NaN : (double) ok / n;
        return new double[]{acc, n};
    }

    /** דיוק לפי טיקר. */
    public static Map<String, Double> accuracyByTicker(String[] tickers, int[] y, int[] yhat) {
        Map<String, int[]> agg = new LinkedHashMap<>();
        for (int i = 0; i < tickers.length; i++) {
            String t = tickers[i];
            agg.putIfAbsent(t, new int[]{0, 0}); // {ok, total}
            int[] a = agg.get(t);
            if (y[i] == yhat[i]) a[0]++;
            a[1]++;
        }
        Map<String, Double> out = new LinkedHashMap<>();
        for (var e : agg.entrySet()) {
            int ok = e.getValue()[0], tot = e.getValue()[1];
            out.put(e.getKey(), tot == 0 ? Double.NaN : (double) ok / tot);
        }
        return out;
    }

    /** שמירת תחזיות ל-CSV. */
    public static void savePredictions(Path path, String[] dates, String[] tickers, int[] y, int[] yhat) throws IOException {
        try (var w = Files.newBufferedWriter(path)) {
            w.write("date,ticker,true_label,pred_label,correct\n");
            for (int i = 0; i < y.length; i++) {
                boolean ok = (y[i] == yhat[i]);
                w.write(dates[i] + "," + tickers[i] + "," + y[i] + "," + yhat[i] + "," + ok + "\n");
            }
        }
    }

    public static String fmtPct(double v) {
        if (Double.isNaN(v)) return "n/a";
        return String.format(Locale.US, "%.1f%%", 100.0 * v);
    }

    /** Confusion Matrix: rows = true labels, cols = predicted labels */
    public static int[][] confusion(int[] y, int[] yhat, int K) {
        int[][] cm = new int[K][K];
        for (int i = 0; i < y.length; i++) {
            if (y[i] >= 0 && y[i] < K && yhat[i] >= 0 && yhat[i] < K) {
                cm[y[i]][yhat[i]]++;
            }
        }
        return cm;
    }

    public static void printConfusion(int[][] cm) {
        int K = cm.length;
        String[] names;

        if (K == 5) {
            names = new String[]{
                    "0=Strong Down",
                    "1=Down",
                    "2=No Change",
                    "3=Up",
                    "4=Strong Up"
            };
        } else if (K == 3) {
            names = new String[]{
                    "0=setosa",
                    "1=versicolor",
                    "2=virginica"
            };
        } else {
            names = new String[K];
            for (int i = 0; i < K; i++) names[i] = String.valueOf(i);
        }

        System.out.println("Confusion Matrix (rows=true, cols=pred):");

        // הדפסת כותרות לעמודות
        System.out.printf("%-14s", "True\\Pred");
        for (int j = 0; j < K; j++) {
            System.out.printf("%12s", names[j]);
        }
        System.out.println();

        // הדפסת שורות עם שם המחלקה
        for (int i = 0; i < K; i++) {
            System.out.printf("%-14s", names[i]);
            for (int j = 0; j < K; j++) {
                System.out.printf("%12d", cm[i][j]);
            }
            System.out.println();
        }
    }



    /** Macro-averaged F1 score */
    public static double macroF1(int[] y, int[] yhat, int K) {
        int[][] cm = confusion(y, yhat, K);
        double sumF1 = 0.0; int count = 0;
        for (int k = 0; k < K; k++) {
            int tp = cm[k][k];
            int fp = 0, fn = 0;
            for (int j = 0; j < K; j++) {
                if (j != k) {
                    fp += cm[j][k];  // predicted k but true j
                    fn += cm[k][j];  // true k but predicted j
                }
            }
            double prec = (tp + fp == 0) ? 0 : (double) tp / (tp + fp);
            double rec  = (tp + fn == 0) ? 0 : (double) tp / (tp + fn);
            double f1   = (prec + rec == 0) ? 0 : 2 * prec * rec / (prec + rec);
            sumF1 += f1; count++;
        }
        return sumF1 / Math.max(1, count);
    }


    public static void savePredictionsSimple(Path path, int[] y, int[] yhat) throws IOException {
        try (var w = Files.newBufferedWriter(path)) {
            w.write("index,true_label,pred_label,correct\n");
            for (int i = 0; i < y.length; i++) {
                boolean ok = (y[i] == yhat[i]);
                w.write(i + "," + y[i] + "," + yhat[i] + "," + ok + "\n");
            }
        }
    }



}
