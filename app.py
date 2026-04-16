# --- MAIN OUTPUT ---
tab1, tab2 = st.tabs(["AI Confidence & Insights", "Benchmarking"])

# =========================
# TAB 1: AI + INSIGHTS
# =========================
with tab1:

    st.subheader("🔍 Model Confidence")

    ordered_labels = ['Low', 'Moderate', 'High']
    label_to_idx = {label: i for i, label in enumerate(labels)}

    for label in ordered_labels:
        idx = label_to_idx[label]
        conf = probs[idx]
        st.progress(conf, text=f"{label}: {conf*100:.1f}%")

    st.divider()
    st.subheader("💡 Strategic Insights")

    high_prob = probs[label_to_idx["High"]]

    if high_prob < 0.4:
        st.warning("Low probability of High productivity output.")

    if incentive < AVERAGES['High']['incentive']:
        st.info("Incentive is below high-performance benchmark. This may limit output.")

    if idle_time > 0:
        st.error("Idle time detected — direct efficiency loss signal.")

# =========================
# TAB 2: BENCHMARKING
# =========================
with tab2:

    st.subheader("📈 Industry Benchmark Comparison")

    def normalize(value, benchmark):
        return min(value / benchmark, 1.5) if benchmark else 0

    metrics = {
        "Task Complexity (SMV)": (smv, AVERAGES['Moderate']['smv']),
        "Workload (WIP)": (wip, AVERAGES['Moderate']['wip']),
        "Incentive Level": (incentive, AVERAGES['High']['incentive']),
        "Workers": (workers, AVERAGES['Moderate']['workers'])
    }

    for name, (val, ref) in metrics.items():
        ratio = normalize(val, ref)

        st.write(f"**{name}**")
        st.caption(f"Value: {val} | Benchmark: {ref}")
        st.progress(min(ratio / 1.5, 1.0))
