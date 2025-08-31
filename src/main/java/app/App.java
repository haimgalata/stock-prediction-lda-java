package app;

import data.DatasetLoader;
import data.DatasetLoader.Dataset;
import eval.Metrics;
import model.LdaClassifier;

import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;

public class App {
    public static void main(String[] args) {
        Locale.setDefault(Locale.US); // הדפסות עקביות
        final String DATA_FOLDER = "src/main/resources/data"; // שימי כאן את ה-CSV-ים
        final int K = 5;                 // 5 קטגוריות
        final double EPSILON = 0.5;      // %± לאזור "כמעט ללא שינוי"
        final double BIGMOVE = 2.0;      // %± לתנועה "גדולה"
        final boolean USE_LOG_VOLUME = true;
        final boolean ADD_DERIVED = true;
        final double REG = 1e-4;         // רגולריזציה ל-Σ

        try {
            // 1) טעינת דאטה + תיוג + פיצול לפי זמן
            DatasetLoader loader = new DatasetLoader();
            Dataset ds = loader.loadFolder(DATA_FOLDER, EPSILON, BIGMOVE, USE_LOG_VOLUME, ADD_DERIVED);

            System.out.println("Loaded. Train=" + ds.Xtrain.length + "  Test=" + ds.Xtest.length);

            // 2) אימון LDA
            LdaClassifier lda = new LdaClassifier(REG);
            lda.fit(ds.Xtrain, ds.ytrain, K);

            // 3) ניבוי
            int[] yhat = lda.predict(ds.Xtest);

            // פיזור תחזיות: לראות אם בכלל מנבאים 0/4
            int[] predDist = new int[K];
            System.out.println("Classes: 0=Strong Down, 1=Down, 2=No Change, 3=Up, 4=Strong Up");
            for (int v : yhat) if (v >= 0 && v < K) predDist[v]++;
            System.out.print("Predicted dist: ");
            for (int k = 0; k < K; k++) System.out.print(k + ":" + predDist[k] + (k + 1 < K ? ", " : ""));
            System.out.println();

            // 4) דוחות קצרים
            // פיזור כיתות בבדיקה
            int[] dist = Metrics.classCounts(ds.ytest, K);
            System.out.print("Class dist (test): ");
            for (int k = 0; k < K; k++) System.out.print(k + ":" + dist[k] + (k + 1 < K ? ", " : ""));
            System.out.println();

            // Baseline majority (על טסט)
            int majority = Metrics.majorityClass(ds.ytest, K);
            int majCount = dist[majority];
            double majAcc = (double) majCount / ds.ytest.length;
            System.out.println("Baseline (majority=" + majority + "): " + Metrics.fmtPct(majAcc));

            // Accuracy כללי
            double acc = Metrics.accuracy(ds.ytest, yhat);
            System.out.println("LDA Accuracy: " + Metrics.fmtPct(acc));

            // Confusion Matrix + Macro-F1
            int[][] cm = Metrics.confusion(ds.ytest, yhat, K);
            Metrics.printConfusion(cm);

            double macroF1 = Metrics.macroF1(ds.ytest, yhat, K);
            System.out.println("Macro-F1: " + Metrics.fmtPct(macroF1));

            // Big-move accuracy (רק כיתות 0 ו-4)
            double[] big = Metrics.accuracyOnLabels(ds.ytest, yhat, 0, 4);
            System.out.println("Big-move accuracy (|ret|>= " + BIGMOVE + "%), n=" + (int)big[1] + ": " + Metrics.fmtPct(big[0]));

            // דיוק לכל כיתה
            double[] perClass = Metrics.perClassHitRate(ds.ytest, yhat, K);
            System.out.print("Per-class hit-rate: {");
            for (int k = 0; k < K; k++) {
                System.out.print(k + ":" + Metrics.fmtPct(perClass[k]) + (k + 1 < K ? ", " : ""));
            }
            System.out.println("}");

            // דיוק לפי טיקר
            Map<String, Double> byTicker = Metrics.accuracyByTicker(ds.tickersTest, ds.ytest, yhat);
            System.out.println("By ticker:");
            for (var e : byTicker.entrySet()) {
                System.out.println("  " + e.getKey() + "  " + Metrics.fmtPct(e.getValue()));
            }

            // 5) שמירת תחזיות ל-CSV
            Path out = Path.of("predictions.csv");
            Metrics.savePredictions(out, ds.datesTest, ds.tickersTest, ds.ytest, yhat);
            System.out.println("Saved: " + out.toAbsolutePath());

        } catch (Exception e) {
            System.err.println("ERROR: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
