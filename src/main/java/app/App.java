package app;

import data.DatasetLoader;
import data.DatasetLoader.Dataset;
import eval.Metrics;
import model.LdaClassifier;

import java.nio.file.Path;
import java.util.*;

public class App {
    public static void main(String[] args) {
        Locale.setDefault(Locale.US); // הדפסות עקביות

        // נתיבים קבועים
        final String DATA_ROOT      = "src/main/resources/data";
        final String STOCKS_FOLDER  = DATA_ROOT + "/stocks";    // רק קבצי מניות
        final String IRIS_CSV_PATH  = DATA_ROOT + "/iris.csv";  // קובץ יחיד

        // פרמטרים למניות
        final double EPSILON = 0.5;      // %± לאזור "כמעט ללא שינוי"
        final double BIGMOVE = 2.0;      // %± לתנועה "גדולה"
        final boolean USE_LOG_VOLUME = true;
        final boolean ADD_DERIVED = true;

        // פרמטרים משותפים
        final double REG = 1e-4;         // רגולריזציה ל-Σ

        // בחירת מצב: דגלים > דיאלוג
        String mode = parseModeFromArgs(args); // "iris" או "stocks"
        if (mode == null) mode = promptModeInteractive();

        try {
            // 1) טעינת דאטה
            DatasetLoader loader = new DatasetLoader();
            Dataset ds;
            int K;

            if ("iris".equals(mode)) {
                ds = loader.loadIris(IRIS_CSV_PATH, 0.7, 42L);
                K = 3; // שלוש מחלקות ב-Iris
                System.out.println("Loaded IRIS. Train=" + ds.Xtrain.length + "  Test=" + ds.Xtest.length);
            } else {
                ds = loader.loadFolder(STOCKS_FOLDER, EPSILON, BIGMOVE, USE_LOG_VOLUME, ADD_DERIVED);
                K = 5; // חמש מחלקות במניות
                System.out.println("Loaded STOCKS. Train=" + ds.Xtrain.length + "  Test=" + ds.Xtest.length);
            }

            // 2) אימון LDA
            LdaClassifier lda = new LdaClassifier(REG);
            lda.fit(ds.Xtrain, ds.ytrain, K);

            // 3) ניבוי
            int[] yhat = lda.predict(ds.Xtest);

            // 4) דוחות משותפים
            int[] predDist = new int[K];
            System.out.println("Classes: " + (K == 5
                    ? "0=Strong Down, 1=Down, 2=No Change, 3=Up, 4=Strong Up"
                    : "0=setosa, 1=versicolor, 2=virginica"));
            for (int v : yhat) if (v >= 0 && v < K) predDist[v]++;
            System.out.print("Predicted dist: ");
            for (int k = 0; k < K; k++) System.out.print(k + ":" + predDist[k] + (k + 1 < K ? ", " : ""));
            System.out.println();

            int[] dist = Metrics.classCounts(ds.ytest, K);
            System.out.print("Class dist (test): ");
            for (int k = 0; k < K; k++) System.out.print(k + ":" + dist[k] + (k + 1 < K ? ", " : ""));
            System.out.println();

            int majority = Metrics.majorityClass(ds.ytest, K);
            double majAcc = (double) dist[majority] / ds.ytest.length;
            System.out.println("Baseline (majority=" + majority + "): " + Metrics.fmtPct(majAcc));

            double acc = Metrics.accuracy(ds.ytest, yhat);
            System.out.println("LDA Accuracy: " + Metrics.fmtPct(acc));

            int[][] cm = Metrics.confusion(ds.ytest, yhat, K);
            Metrics.printConfusion(cm);

            double macroF1 = Metrics.macroF1(ds.ytest, yhat, K);
            System.out.println("Macro-F1: " + Metrics.fmtPct(macroF1));

            double[] perClass = Metrics.perClassHitRate(ds.ytest, yhat, K);
            System.out.print("Per-class hit-rate: {");
            for (int k = 0; k < K; k++) {
                System.out.print(k + ":" + Metrics.fmtPct(perClass[k]) + (k + 1 < K ? ", " : ""));
            }
            System.out.println("}");

            // 5) דוחות ייחודיים למניות
            if ("stocks".equals(mode)) {
                double[] big = Metrics.accuracyOnLabels(ds.ytest, yhat, 0, 4);
                System.out.println("Big-move accuracy (|ret|>= 2.0%), n=" + (int)big[1] + ": " + Metrics.fmtPct(big[0]));

                Map<String, Double> byTicker = Metrics.accuracyByTicker(ds.tickersTest, ds.ytest, yhat);
                System.out.println("By ticker:");
                for (var e : byTicker.entrySet()) {
                    System.out.println("  " + e.getKey() + "  " + Metrics.fmtPct(e.getValue()));
                }
            }

            // 6) שמירת תחזיות
            Path out = Path.of("stocks".equals(mode) ? "predictions.csv" : "predictions_iris.csv");
            if ("stocks".equals(mode)) {
                Metrics.savePredictions(out, ds.datesTest, ds.tickersTest, ds.ytest, yhat);
            } else {
                Metrics.savePredictionsSimple(out, ds.ytest, yhat);
            }
            System.out.println("Saved: " + out.toAbsolutePath());

        } catch (Exception e) {
            System.err.println("ERROR: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // --- עזרים: בחירת מצב ---
    private static String parseModeFromArgs(String[] args) {
        if (args == null) return null;
        for (String a : args) {
            if (a == null) continue;
            String s = a.trim().toLowerCase(Locale.ROOT);
            if (s.equals("--iris"))   return "iris";
            if (s.equals("--stocks") || s.equals("--stock")) return "stocks";
        }
        return null;
    }

    private static String promptModeInteractive() {
        try {
            Scanner sc = new Scanner(System.in);
            while (true) {
                System.out.println("Choose dataset to run: [1] IRIS  |  [2] STOCKS");
                System.out.print("Type 1 or 2 and press Enter: ");
                String s = sc.nextLine().trim().toLowerCase(Locale.ROOT);
                if (s.equals("1") || s.startsWith("i") || s.contains("iris")) return "iris";
                if (s.equals("2") || s.startsWith("s") || s.contains("stock")) return "stocks";
                System.out.println("קלט לא תקין. נסה שוב.");
            }
        } catch (Exception ignored) {
            System.out.println("אין קלט אינטראקטיבי – מצב ברירת מחדל: STOCKS.");
            return "stocks";
        }
    }
}
