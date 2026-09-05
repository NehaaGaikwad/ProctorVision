import streamlit as st
import sqlite3
import os
import json
import pandas as pd


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATABASE = os.path.join(
    PROJECT_ROOT,
    "database",
    "proctorvision.db"
)

REPORTS_DIR = os.path.join(
    PROJECT_ROOT,
    "reports"
)


st.set_page_config(
    page_title="ProctorVision AI",
    page_icon="🛡️",
    layout="wide"
)


def get_connection():
    return sqlite3.connect(DATABASE)


def get_sessions():

    connection = get_connection()

    dataframe = pd.read_sql_query(
        """
        SELECT
            session_id,
            start_time,
            end_time,
            status
        FROM sessions
        ORDER BY start_time DESC
        """,
        connection
    )

    connection.close()

    return dataframe


def get_violations():

    connection = get_connection()

    dataframe = pd.read_sql_query(
        """
        SELECT
            id,
            type,
            timestamp,
            duration,
            confidence,
            severity,
            evidence_path,
            session_id
        FROM violations
        ORDER BY timestamp ASC
        """,
        connection
    )

    connection.close()

    return dataframe


def get_session_violations(session_id):

    connection = get_connection()

    dataframe = pd.read_sql_query(
        """
        SELECT
            id,
            type,
            timestamp,
            duration,
            confidence,
            severity,
            evidence_path
        FROM violations
        WHERE session_id = ?
        ORDER BY timestamp ASC
        """,
        connection,
        params=(session_id,)
    )

    connection.close()

    return dataframe


def calculate_risk_score(dataframe):

    score = 0

    for _, violation in dataframe.iterrows():

        if violation["severity"] == "HIGH":
            score += 10

        elif violation["severity"] == "MEDIUM":
            score += 5

        elif violation["severity"] == "LOW":
            score += 2

    return min(score, 100)


def get_final_result(dataframe):

    if dataframe.empty:
        return "NO VIOLATIONS"

    if "HIGH" in dataframe["severity"].values:
        return "HIGH RISK"

    if "MEDIUM" in dataframe["severity"].values:
        return "MEDIUM RISK"

    return "LOW RISK"


def get_report_path(session_id):

    return os.path.join(
        REPORTS_DIR,
        f"{session_id}.json"
    )


st.title("🛡️ ProctorVision AI")

st.caption(
    "AI-Based Online Exam Proctoring & Violation Analytics"
)


if st.button("🔄 Refresh Dashboard"):

    st.rerun()


st.divider()


if not os.path.exists(DATABASE):

    st.error(
        "Database not found. Run the ProctorVision exam system first."
    )

    st.stop()


sessions = get_sessions()
violations = get_violations()


total_sessions = len(sessions)
total_violations = len(violations)


high_count = len(
    violations[
        violations["severity"] == "HIGH"
    ]
)


medium_count = len(
    violations[
        violations["severity"] == "MEDIUM"
    ]
)


low_count = len(
    violations[
        violations["severity"] == "LOW"
    ]
)


st.markdown("## 📊 Overall Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Sessions",
        total_sessions
    )


with col2:

    st.metric(
        "Total Violations",
        total_violations
    )


with col3:

    st.metric(
        "High Severity",
        high_count
    )


with col4:

    st.metric(
        "Medium Severity",
        medium_count
    )


st.divider()


st.markdown("## 📈 Violation Analytics")


if violations.empty:

    st.info(
        "No violations available for analytics."
    )

else:

    col1, col2 = st.columns(2)


    with col1:

        st.markdown("### Violation Types")


        type_data = (
            violations["type"]
            .value_counts()
            .rename_axis("Type")
            .reset_index(name="Count")
        )


        st.bar_chart(
            type_data.set_index("Type")
        )


    with col2:

        st.markdown("### Severity Distribution")


        severity_data = pd.DataFrame(
            {
                "Severity": [
                    "HIGH",
                    "MEDIUM",
                    "LOW"
                ],
                "Count": [
                    high_count,
                    medium_count,
                    low_count
                ]
            }
        )


        st.bar_chart(
            severity_data.set_index("Severity")
        )


st.divider()


st.markdown("## 👤 Face Monitoring Analytics")


no_face_count = 0
multiple_face_count = 0


if not violations.empty:

    violation_types = (
        violations["type"]
        .astype(str)
        .str.upper()
    )


    no_face_count = (
        violation_types == "NO_FACE"
    ).sum()


    multiple_face_count = (
        violation_types == "MULTIPLE_FACES"
    ).sum()


face_data = pd.DataFrame(
    {
        "Detection": [
            "No Face",
            "Multiple Faces"
        ],
        "Count": [
            no_face_count,
            multiple_face_count
        ]
    }
)


st.bar_chart(
    face_data.set_index("Detection")
)


st.divider()


st.markdown("## 🎯 Exam Session Analysis")


if sessions.empty:

    st.info(
        "No exam sessions available."
    )

else:

    selected_session = st.selectbox(
        "Select Exam Session",
        sessions["session_id"].tolist()
    )


    session_data = sessions[
        sessions["session_id"] == selected_session
    ].iloc[0]


    session_violations = get_session_violations(
        selected_session
    )


    risk_score = calculate_risk_score(
        session_violations
    )


    final_result = get_final_result(
        session_violations
    )


    if (
        pd.notna(session_data["start_time"])
        and
        pd.notna(session_data["end_time"])
    ):

        start_time = pd.to_datetime(
            session_data["start_time"]
        )


        end_time = pd.to_datetime(
            session_data["end_time"]
        )


        duration = (
            end_time - start_time
        ).total_seconds()

    else:

        duration = 0


    col1, col2, col3, col4, col5 = st.columns(
        [1, 1, 1, 1.5, 1]
    )


    with col1:

        st.metric(
            "Status",
            session_data["status"]
        )


    with col2:

        st.metric(
            "Violations",
            len(session_violations)
        )


    with col3:

        st.metric(
            "Risk Score",
            f"{risk_score}/100"
        )


    with col4:

        st.markdown(
            """
            <style>
            .result-value {
                font-size: 2rem;
                font-weight: 400;
                line-height: 1.2;
                white-space: nowrap;
            }
            </style>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            "Result",
            unsafe_allow_html=False
        )


        st.markdown(
            f'<div class="result-value">{final_result}</div>',
            unsafe_allow_html=True
        )


    with col5:

        st.metric(
            "Duration",
            f"{duration:.0f}s"
        )


    st.write(
        f"**Session ID:** {selected_session}"
    )


    st.write(
        f"**Start Time:** {session_data['start_time']}"
    )


    st.write(
        f"**End Time:** {session_data['end_time']}"
    )


    st.divider()


    st.markdown("### Session Violation Breakdown")


    if session_violations.empty:

        st.success(
            "No violations detected in this session."
        )

    else:

        col1, col2 = st.columns(2)


        with col1:

            session_type_data = (
                session_violations["type"]
                .value_counts()
                .rename_axis("Type")
                .reset_index(name="Count")
            )


            st.bar_chart(
                session_type_data.set_index("Type")
            )


        with col2:

            session_severity_data = (
                session_violations["severity"]
                .value_counts()
                .rename_axis("Severity")
                .reset_index(name="Count")
            )


            st.bar_chart(
                session_severity_data.set_index(
                    "Severity"
                )
            )


    st.divider()


    st.markdown("### 🕐 Session Timeline")


    if not session_violations.empty:

        timeline_data = session_violations.copy()


        timeline_data["timestamp"] = pd.to_datetime(
            timeline_data["timestamp"]
        )


        timeline_data["time"] = (
            timeline_data["timestamp"]
            .dt.strftime("%H:%M:%S")
        )


        timeline_data = timeline_data[
            [
                "time",
                "type",
                "severity",
                "confidence",
                "duration"
            ]
        ].copy()


        timeline_data["confidence"] = (
            timeline_data["confidence"] * 100
        ).round(2)


        timeline_data["duration"] = (
            timeline_data["duration"]
            .fillna(0)
            .round(2)
        )


        timeline_data = timeline_data.rename(
            columns={
                "time": "Time",
                "type": "Type",
                "severity": "Severity",
                "confidence": "Confidence (%)",
                "duration": "Duration (s)"
            }
        )


        st.dataframe(
            timeline_data,
            use_container_width=True,
            hide_index=True
        )


    st.divider()


    st.markdown("### 🚨 Violation Details")


    if not session_violations.empty:

        violation_options = []


        for _, violation in session_violations.iterrows():

            label = (
                f"#{int(violation['id'])} | "
                f"{violation['type']} | "
                f"{violation['timestamp']} | "
                f"{violation['severity']}"
            )


            violation_options.append(
                (
                    label,
                    int(violation["id"])
                )
            )


        selected_label = st.selectbox(
            "Select Violation",
            [
                item[0]
                for item in violation_options
            ]
        )


        selected_id = next(
            item[1]
            for item in violation_options
            if item[0] == selected_label
        )


        selected_violation = session_violations[
            session_violations["id"] == selected_id
        ].iloc[0]


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Type",
                selected_violation["type"]
            )


        with col2:

            confidence = (
                float(
                    selected_violation["confidence"]
                ) * 100
            )


            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


        with col3:

            violation_duration = (
                selected_violation["duration"]
            )


            if pd.isna(
                violation_duration
            ):

                violation_duration = 0


            st.metric(
                "Duration",
                f"{float(violation_duration):.2f}s"
            )


        with col4:

            st.metric(
                "Severity",
                selected_violation["severity"]
            )


        st.write(
            f"**Timestamp:** "
            f"{selected_violation['timestamp']}"
        )


        evidence_path = (
            selected_violation["evidence_path"]
        )


        st.markdown("#### Evidence")


        if (
            pd.isna(evidence_path)
            or
            not evidence_path
        ):

            st.info(
                "No evidence available."
            )

        else:

            evidence_path = str(
                evidence_path
            )


            if not os.path.exists(
                evidence_path
            ):

                st.warning(
                    "Evidence file not found."
                )


                st.code(
                    evidence_path
                )

            else:

                violation_type = (
                    selected_violation["type"]
                ).upper()


                if violation_type in [
                    "PHONE",
                    "BOOK",
                    "WINDOW_SWITCH"
                ]:

                    st.image(
                        evidence_path,
                        caption=(
                            f"{violation_type} Evidence"
                        ),
                        use_container_width=True
                    )


                elif violation_type == "VOICE":

                    st.audio(
                        evidence_path
                    )


                    st.write(
                        os.path.basename(
                            evidence_path
                        )
                    )


    st.divider()


    st.markdown("### 📄 Session Report")


    report_path = get_report_path(
        selected_session
    )


    if os.path.exists(report_path):

        with open(
            report_path,
            "r",
            encoding="utf-8"
        ) as file:

            report_data = json.load(file)


        st.download_button(
            label="⬇️ Download JSON Report",
            data=json.dumps(
                report_data,
                indent=4
            ),
            file_name=(
                f"{selected_session}.json"
            ),
            mime="application/json"
        )


        with st.expander(
            "View JSON Report"
        ):

            st.json(
                report_data
            )

    else:

        st.warning(
            "JSON report not found for this session."
        )


st.divider()


st.markdown("## 📋 All Exam Sessions")


if sessions.empty:

    st.info(
        "No sessions available."
    )

else:

    sessions_display = sessions.copy()


    sessions_display = sessions_display.rename(
        columns={
            "session_id": "Session ID",
            "start_time": "Start Time",
            "end_time": "End Time",
            "status": "Status"
        }
    )


    st.dataframe(
        sessions_display,
        use_container_width=True,
        hide_index=True
    )