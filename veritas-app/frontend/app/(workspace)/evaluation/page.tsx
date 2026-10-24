import styles from "../overview/overview.module.css";

export default function EvaluationPage() {
  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>Evaluation</h1>
          <p className={styles.subtitle}>Quantitative metrics and qualitative results</p>
        </div>
      </div>
      <div className={styles.section}>
        <p style={{ color: "#75787d", fontSize: "0.9rem" }}>Coming soon — evaluation metrics and comparison tables.</p>
      </div>
    </div>
  );
}
