package model;

import util.MathUtils;
import java.util.Arrays;

/** מימוש LDA מאפס: ממוצעים לפי כיתה, קו-וריאנצה משותפת, וחוק החלטה לינארי. */
public class LdaClassifier {
    private final double reg;        // רגולריזציה אלכסונית ל-Σ
    private int K;                   // מספר כיתות
    private int d;                   // מספר פיצ'רים
    private double[][] mu;           // ממוצע לכל כיתה [K][d]
    private double[] pi;             // priors לכל כיתה [K]
    private double[][] sigma;        // קו-וריאנצה משותפת [d][d]
    private double[][] w;            // w_k = Σ^{-1} μ_k  [K][d]
    private double[] c;              // הטייה: c_k = -1/2 μ_k^T w_k + log π_k

    /** @param reg רגולריזציה אלכסונית (למשל 1e-6). */
    public LdaClassifier(double reg) {
        this.reg = reg;
    }

    /** אימון: מחשב μ_k, Σ, π_k ואז w_k ו-c_k. */
    public void fit(double[][] X, int[] y, int numClasses) {
        int N = X.length;
        if (N == 0) throw new IllegalArgumentException("Empty dataset");
        d = X[0].length;
        for (double[] row : X) if (row.length != d) throw new IllegalArgumentException("Inconsistent dims");

        this.K = numClasses;
        mu = new double[K][d];
        pi = new double[K];
        int[] count = new int[K];

        // ממוצעים לפי כיתה
        for (int i = 0; i < N; i++) {
            int k = y[i];
            if (k < 0 || k >= K) throw new IllegalArgumentException("Label out of range: " + k);
            count[k]++;
            double[] xi = X[i];
            for (int j = 0; j < d; j++) mu[k][j] += xi[j];
        }
        for (int k = 0; k < K; k++) {
            if (count[k] == 0) throw new IllegalArgumentException("Empty class k=" + k);
            for (int j = 0; j < d; j++) mu[k][j] /= count[k];
            pi[k] = 1.0 / K;
        }

        // קו-וריאנצה משותפת (pooled): Σ = 1/(N-K) * Σ_k Σ_{i∈Ck} (x - μ_k)(x - μ_k)^T
        sigma = new double[d][d];
        for (int i = 0; i < N; i++) {
            int k = y[i];
            double[] xi = X[i];
            double[] mk = mu[k];
            for (int a = 0; a < d; a++) {
                double da = xi[a] - mk[a];
                for (int b = 0; b < d; b++) {
                    sigma[a][b] += da * (xi[b] - mk[b]);
                }
            }
        }
        double denom = Math.max(1, N - K);
        for (int a = 0; a < d; a++)
            for (int b = 0; b < d; b++)
                sigma[a][b] /= denom;

        // רגולריזציה ליציבות
        double trace = 0.0;
        for (int i = 0; i < d; i++) trace += sigma[i][i];
        double lambda = reg * (trace / Math.max(1, d));
        MathUtils.addToDiagonal(sigma, lambda);

        // w_k = Σ^{-1} μ_k  ו-  c_k = -1/2 μ_k^T w_k + log π_k
        w = new double[K][d];
        c = new double[K];
        for (int k = 0; k < K; k++) {
            w[k] = MathUtils.solveLinearSystem(copy2D(sigma), mu[k]); // פותרים Σ w = μ
            double muTw = MathUtils.dot(mu[k], w[k]);
            c[k] = -0.5 * muTw + MathUtils.safeLog(pi[k]);
        }
    }

    /** ציון/דיסקרימיננטה לכל כיתה עבור x. */
    private double[] scores(double[] x) {
        double[] s = new double[K];
        for (int k = 0; k < K; k++) {
            s[k] = MathUtils.dot(w[k], x) + c[k]; // δ_k(x) = x^T w_k + c_k
        }
        return s;
    }

    /** ניבוי בודד: argmax_k δ_k(x). */
    public int predictOne(double[] x) {
        double[] s = scores(x);
        int arg = 0; double best = s[0];
        for (int k = 1; k < K; k++) if (s[k] > best) { best = s[k]; arg = k; }
        return arg;
    }

    /** ניבוי למטריצה שלמה. */
    public int[] predict(double[][] X) {
        int[] yhat = new int[X.length];
        for (int i = 0; i < X.length; i++) yhat[i] = predictOne(X[i]);
        return yhat;
    }

    /** הסתברויות (Softmax של הציונים; הערכה נוחה לדוחות). */
    public double[] predictProba(double[] x) {
        double[] s = scores(x);
        double max = Arrays.stream(s).max().orElse(0.0);
        double sum = 0.0;
        for (int k = 0; k < K; k++) { s[k] = Math.exp(s[k] - max); sum += s[k]; }
        for (int k = 0; k < K; k++) s[k] /= sum;
        return s;
    }

    // עותק עמוק קטן למטריצה (כדי לא להרוס את Σ כשפותרים)
    private static double[][] copy2D(double[][] A) {
        double[][] B = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) System.arraycopy(A[i], 0, B[i], 0, A[i].length);
        return B;
    }
}
