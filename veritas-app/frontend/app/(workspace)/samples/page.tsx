import styles from "../overview/overview.module.css";

export default function SamplesPage() {
  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>Samples</h1>
          <p className={styles.subtitle}>Sample predictions and visualizations</p>
        </div>
      </div>
      <div className={styles.section}>
        <p style={{ color: "#75787d", fontSize: "0.9rem" }}>Coming soon — sample gallery.</p>
      </div>
    </div>
  );
}
