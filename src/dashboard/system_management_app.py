"""System Management — users, logs, configuration, and system health.

Extracted from the Admin console to provide a focused, single-purpose
page for all system-related administrative tasks.
"""

import streamlit as st
from src.dashboard.theme import page_header


def page_system_management():
    """Renders the System Management page."""
    st.markdown(page_header("⚙️ System Management"), unsafe_allow_html=True)
    st.caption(
        "Manage users, view system logs, review configuration, and check system health. "
        "For pipeline actions and database operations, see the Data Management page."
    )

    tab_status, tab_users, tab_config, tab_logs = st.tabs([
        "ℹ️ System Status", "👤 Users", "📝 Configuration", "📜 Logs"
    ])

    with tab_status:
        from src.dashboard.app import _admin_status_tab
        _admin_status_tab()
        # FinBERT Cache Health
        st.markdown("#### FinBERT Cache Health")
        try:
            from src.data.db_router import get_router
            import yaml
            with open("config.yaml") as _f:
                _cfg = yaml.safe_load(_f) or {}
            _router = get_router(_cfg)
            cache_df = _router.query(
                "SELECT COUNT(*) as total_cached, "
                "AVG(fb_score) as avg_score, "
                "MIN(scored_at) as oldest, "
                "MAX(scored_at) as newest "
                "FROM finbert_cache"
            )
            if not cache_df.empty and cache_df.iloc[0]["total_cached"]:
                row = cache_df.iloc[0]
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    st.metric("Total Cached", f"{int(row['total_cached'] or 0):,}")
                with cc2:
                    avg = float(row["avg_score"] or 0)
                    st.metric("Avg Sentiment", f"{avg:+.3f}")
                with cc3:
                    st.caption(
                        f"Range: {(row.get('oldest') or 'N/A')[:19]} → "
                        f"{(row.get('newest') or 'N/A')[:19]}"
                    )
            else:
                st.info("finbert_cache is empty — will populate on next pipeline run")
        except Exception as e:
            st.warning(f"Could not read finbert_cache: {e}")

    with tab_users:
        from src.dashboard.app import _admin_users_tab
        _admin_users_tab()

    with tab_config:
        from src.dashboard.app import _admin_config_tab
        _admin_config_tab()

    with tab_logs:
        from src.dashboard.app import _admin_logs_tab
        _admin_logs_tab()
