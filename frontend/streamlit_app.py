"""ConnectTel Customer Churn Predictor - Streamlit Frontend Application.

Production-grade AI web application providing customer retention risk assessment
by communicating with the ConnectTel FastAPI backend service.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

import streamlit as st

# Ensure capstone root is on sys.path so frontend can import api_client
FRONTEND_DIR = Path(__file__).resolve().parent
CAPSTONE_DIR = FRONTEND_DIR.parent
if str(CAPSTONE_DIR) not in sys.path:
    sys.path.insert(0, str(CAPSTONE_DIR))

from frontend.api_client import check_health, predict_churn, get_backend_url


# Set Page Configuration
st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern SaaS aesthetic
CUSTOM_CSS = """
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Risk Tier Banners */
    .risk-banner {
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1.25rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .risk-high {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        color: #991b1b;
    }
    .risk-medium {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        color: #92400e;
    }
    .risk-low {
        background-color: #f0fdf4;
        border-left: 5px solid #10b981;
        color: #065f46;
    }

    /* Recommendation Cards */
    .rec-item {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    .rec-icon {
        font-size: 1.25rem;
        line-height: 1;
    }
    .rec-text {
        font-size: 0.9rem;
        color: #334155;
    }

    /* Form section headers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Default Sample Customer Profiles for Quick Testing
PROFILE_PRESETS = {
    "High Risk (Month-to-Month, Fiber Optic, No Add-ons)": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 239.55,
    },
    "Low Risk (Long-Term Loyal, 2-Year Contract, Protected)": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 60.50,
        "TotalCharges": 3932.50,
    },
    "Medium Risk (Mid-Tenure, 1-Year Contract, Moderate Add-ons)": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 94.00,
        "TotalCharges": 2256.00,
    },
}


def initialize_session_state():
    """Ensure session state variables exist."""
    if "current_profile" not in st.session_state:
        st.session_state.current_profile = PROFILE_PRESETS[
            "High Risk (Month-to-Month, Fiber Optic, No Add-ons)"
        ].copy()
    if "latest_prediction" not in st.session_state:
        st.session_state.latest_prediction = None
    if "submitted_profile" not in st.session_state:
        st.session_state.submitted_profile = None


initialize_session_state()


# Sidebar: Server Status & Quick Presets
with st.sidebar:
    st.title("Settings & Presets")

    # Backend Connection Status
    backend_url = get_backend_url()
    st.caption(f"Backend Target: `{backend_url}`")
    health = check_health(backend_url)

    if health["is_available"] and health["model_loaded"]:
        st.success(
            f"● Backend Connected\n\n**Model:** {health.get('model_name', 'Active')}\n\n**API Ver:** {health.get('version', '1.0')}"
        )
    elif health["is_available"]:
        st.warning("● Backend Online (Model Initializing...)")
    else:
        st.error(
            f"● Backend Offline\n\n{health.get('error_message', 'Could not reach server.')}"
        )
        st.info("Ensure the FastAPI backend is running via:\n`uvicorn app.main:app --port 8000`")

    st.markdown("---")
    st.subheader("Quick Profile Presets")
    st.caption("Load verified customer personas to test predictions:")

    selected_preset = st.selectbox(
        "Select Demo Profile",
        options=list(PROFILE_PRESETS.keys()),
        index=0,
    )

    if st.button("Load Selected Profile", use_container_width=True):
        st.session_state.current_profile = PROFILE_PRESETS[selected_preset].copy()
        st.session_state.latest_prediction = None
        st.rerun()

    if st.button("Reset Form to Defaults", use_container_width=True):
        st.session_state.current_profile = PROFILE_PRESETS[
            "High Risk (Month-to-Month, Fiber Optic, No Add-ons)"
        ].copy()
        st.session_state.latest_prediction = None
        st.session_state.submitted_profile = None
        st.rerun()

    st.markdown("---")
    st.caption(
        "ConnectTel Predictive Customer Churn Service  \n"
        "Deployment Pipeline: Phase 2 Streamlit UI"
    )


# Main Header
st.title("Telecom Customer Churn Predictor")
st.subheader("AI-powered customer retention risk assessment")
st.markdown(
    "Enter customer demographics, subscription details, and billing history below to generate real-time churn probability and prescriptive retention guidance."
)


# Main Form Container
current = st.session_state.current_profile

with st.form(key="churn_prediction_form"):
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # Group A: Customer Demographics
        st.markdown('<div class="section-header">A. Customer Demographics</div>', unsafe_allow_html=True)
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            gender = st.selectbox(
                "Gender",
                options=["Female", "Male"],
                index=0 if current.get("gender") == "Female" else 1,
                help="Customer biological gender.",
            )
            senior_citizen_val = st.selectbox(
                "Senior Citizen",
                options=[0, 1],
                format_func=lambda x: "Yes (65+)" if x == 1 else "No",
                index=int(current.get("SeniorCitizen", 0)),
                help="Indicates if customer is aged 65 or older.",
            )
        with d_col2:
            partner = st.selectbox(
                "Has Partner",
                options=["Yes", "No"],
                index=0 if current.get("Partner") == "Yes" else 1,
                help="Indicates if the customer lives with a partner.",
            )
            dependents = st.selectbox(
                "Has Dependents",
                options=["Yes", "No"],
                index=0 if current.get("Dependents") == "Yes" else 1,
                help="Indicates if the customer has children or dependents.",
            )

        # Group B: Account & Contract Terms
        st.markdown('<div class="section-header">B. Account & Contract Terms</div>', unsafe_allow_html=True)
        ac_col1, ac_col2 = st.columns(2)
        with ac_col1:
            tenure = st.slider(
                "Tenure (Months)",
                min_value=0,
                max_value=72,
                value=int(current.get("tenure", 12)),
                help="Number of months customer has been subscribed to ConnectTel.",
            )
            contract = st.selectbox(
                "Contract Commitment",
                options=["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(
                    current.get("Contract", "Month-to-month")
                ),
                help="Current contract duration term.",
            )
        with ac_col2:
            paperless = st.selectbox(
                "Paperless Billing",
                options=["Yes", "No"],
                index=0 if current.get("PaperlessBilling") == "Yes" else 1,
                help="Whether paperless invoicing is activated.",
            )
            payment_methods = [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
            payment_method = st.selectbox(
                "Payment Method",
                options=payment_methods,
                index=payment_methods.index(
                    current.get("PaymentMethod", "Electronic check")
                ),
                help="Selected invoice payment method.",
            )

        # Group C: Phone & Internet Core Services
        st.markdown('<div class="section-header">C. Phone & Internet Core Services</div>', unsafe_allow_html=True)
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            phone_service = st.selectbox(
                "Phone Service",
                options=["Yes", "No"],
                index=0 if current.get("PhoneService") == "Yes" else 1,
                help="Whether customer has a landline phone subscription.",
            )
            multiple_lines_opts = ["No", "Yes", "No phone service"]
            multiple_lines = st.selectbox(
                "Multiple Lines",
                options=multiple_lines_opts,
                index=multiple_lines_opts.index(
                    current.get("MultipleLines", "No")
                ),
                help="Multiple phone line subscription status.",
            )
        with s_col2:
            internet_opts = ["DSL", "Fiber optic", "No"]
            internet_service = st.selectbox(
                "Internet Service Provider",
                options=internet_opts,
                index=internet_opts.index(
                    current.get("InternetService", "Fiber optic")
                ),
                help="Primary internet connectivity technology.",
            )

    with col_right:
        # Group D: Security & Technical Support Add-ons
        st.markdown('<div class="section-header">D. Security & Technical Support Add-ons</div>', unsafe_allow_html=True)
        sec_col1, sec_col2 = st.columns(2)
        addon_opts = ["No", "Yes", "No internet service"]

        with sec_col1:
            online_security = st.selectbox(
                "Online Security",
                options=addon_opts,
                index=addon_opts.index(current.get("OnlineSecurity", "No")),
                help="Antivirus and cybersecurity package.",
            )
            online_backup = st.selectbox(
                "Online Backup",
                options=addon_opts,
                index=addon_opts.index(current.get("OnlineBackup", "No")),
                help="Cloud storage and automated backup add-on.",
            )
        with sec_col2:
            device_protection = st.selectbox(
                "Device Protection",
                options=addon_opts,
                index=addon_opts.index(current.get("DeviceProtection", "No")),
                help="Hardware warranty and device insurance.",
            )
            tech_support = st.selectbox(
                "Tech Support",
                options=addon_opts,
                index=addon_opts.index(current.get("TechSupport", "No")),
                help="Dedicated 24/7 technical assistance.",
            )

        # Group E: Entertainment Add-ons
        st.markdown('<div class="section-header">E. Entertainment Streaming Add-ons</div>', unsafe_allow_html=True)
        ent_col1, ent_col2 = st.columns(2)
        with ent_col1:
            streaming_tv = st.selectbox(
                "Streaming TV",
                options=addon_opts,
                index=addon_opts.index(current.get("StreamingTV", "No")),
                help="IPTV streaming channels.",
            )
        with ent_col2:
            streaming_movies = st.selectbox(
                "Streaming Movies",
                options=addon_opts,
                index=addon_opts.index(current.get("StreamingMovies", "No")),
                help="On-demand cinema streaming package.",
            )

        # Group F: Billing History
        st.markdown('<div class="section-header">F. Billing History</div>', unsafe_allow_html=True)
        bill_col1, bill_col2 = st.columns(2)
        with bill_col1:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=15.0,
                max_value=300.0,
                value=float(current.get("MonthlyCharges", 70.0)),
                step=0.50,
                format="%.2f",
                help="Current recurring monthly bill in USD.",
            )
        with bill_col2:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=15000.0,
                value=float(current.get("TotalCharges", 150.0)),
                step=5.0,
                format="%.2f",
                help="Cumulative billing amount since account creation.",
            )

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(
        label="Predict Churn Risk",
        type="primary",
        use_container_width=True,
    )


# Handle Form Submission
if submit_button:
    # Construct exact 19-feature payload conforming to backend schema
    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen_val,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    with st.spinner("Connecting to inference engine..."):
        resp = predict_churn(payload, backend_url=backend_url)

    if resp["success"]:
        st.session_state.latest_prediction = resp["data"]
        st.session_state.submitted_profile = payload
    else:
        st.error(f"Inference Failed: {resp['error_message']}")


# Display Prediction Dashboard
if st.session_state.latest_prediction:
    pred_data = st.session_state.latest_prediction
    prof = st.session_state.submitted_profile or {}

    raw_prob = pred_data.get("churn_probability", 0.0)
    risk_level = pred_data.get("risk_level", "Unknown")
    prediction_label = pred_data.get("churn_label", "No")
    is_churn = pred_data.get("prediction", 0) == 1

    st.markdown("---")
    st.subheader("Assessment Results & Recommendations")

    # Metrics Summary Row
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        status_text = "Likely to Churn" if is_churn else "Likely to Retain"
        status_color = "#dc2626" if is_churn else "#16a34a"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model Verdict</div>
                <div class="metric-value" style="color: {status_color};">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        prob_pct = raw_prob * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Churn Probability</div>
                <div class="metric-value">{prob_pct:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        risk_color = (
            "#dc2626"
            if risk_level == "High Risk"
            else "#d97706"
            if risk_level == "Medium Risk"
            else "#16a34a"
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Risk Tier</div>
                <div class="metric-value" style="color: {risk_color};">{risk_level}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Probability Progress Bar Visualization
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"**Estimated Churn Likelihood: {prob_pct:.2f}%**")
    st.progress(min(max(float(raw_prob), 0.0), 1.0))

    col_b1, col_b2, col_b3 = st.columns([4, 3, 3])
    with col_b1:
        st.caption("🟢 Low Risk (0% – 40%)")
    with col_b2:
        st.caption("🟡 Medium Risk (40% – 70%)")
    with col_b3:
        st.caption("🔴 High Risk (70% – 100%)")

    # Risk Tier Visual Banner
    if risk_level == "High Risk":
        st.markdown(
            f"""
            <div class="risk-banner risk-high">
                <strong>⚠️ HIGH CHURN RISK DETECTED ({prob_pct:.2f}% Likelihood)</strong><br>
                This customer profile exhibits critical churn markers. Immediate proactive intervention by the Customer Retention Team is strongly advised before the next billing cycle.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif risk_level == "Medium Risk":
        st.markdown(
            f"""
            <div class="risk-banner risk-medium">
                <strong>⚡ MODERATE CHURN RISK ({prob_pct:.2f}% Likelihood)</strong><br>
                Customer displays developing friction patterns or low service entrenchment. Targeted relationship nurturing and service review are suggested to prevent escalation.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="risk-banner risk-low">
                <strong>✅ LOW CHURN RISK ({prob_pct:.2f}% Likelihood)</strong><br>
                Account demonstrates healthy loyalty attributes. Maintain standard service excellence and monitor satisfaction routinely.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Actionable Business Retention Recommendations
    st.subheader("Recommended Retention Actions")
    st.caption("Prescriptive business interventions tailored to customer contract, service, and tenure profile (suggested actions, not guaranteed outcomes):")

    recommendations = []

    # Recommendation rules based on entered profile and risk tier
    if risk_level == "High Risk":
        recommendations.append(
            ("📞", "**Priority Outreach:** Trigger immediate outbound contact from a Senior Customer Success Specialist to review account satisfaction.")
        )
        if prof.get("Contract") == "Month-to-month":
            recommendations.append(
                ("📝", "**Contract Transition Incentive:** Propose a 1-year loyalty agreement with a guaranteed 15% rate reduction or complimentary service upgrade.")
            )
        if prof.get("InternetService") == "Fiber optic" and (prof.get("TechSupport") != "Yes" or prof.get("OnlineSecurity") != "Yes"):
            recommendations.append(
                ("🛡️", "**Complimentary Protection Bundle:** Offer 6 months of free Tech Support and Online Security to safeguard high-speed experience and resolve connectivity friction.")
            )
        if prof.get("PaymentMethod") == "Electronic check":
            recommendations.append(
                ("💳", "**Auto-Pay Bill Credit:** Offer a one-time $25 bill credit to switch from Electronic Check to recurring Bank Transfer or Credit Card.")
            )
        if prof.get("tenure", 0) <= 12:
            recommendations.append(
                ("🚀", "**Onboarding Health Check:** Customer is in their first-year vulnerability window. Conduct an executive onboarding satisfaction audit.")
            )

    elif risk_level == "Medium Risk":
        recommendations.append(
            ("📊", "**Quarterly Usage Review:** Schedule an automated service utilization email summarizing monthly value delivered.")
        )
        if prof.get("Contract") == "Month-to-month":
            recommendations.append(
                ("🎁", "**Mid-Term Contract Offer:** Present a 1-year contract renewal package with streaming add-on discounts.")
            )
        if prof.get("OnlineBackup") != "Yes" or prof.get("DeviceProtection") != "Yes":
            recommendations.append(
                ("☁️", "**Value-Add Service Upsell:** Introduce discounted Online Backup or Device Protection to enhance product stickiness.")
            )
        if prof.get("MonthlyCharges", 0) > 80.0:
            recommendations.append(
                ("💰", "**Plan Optimization:** Conduct a package audit to verify whether features match actual usage, mitigating price sensitivity.")
            )

    else:  # Low Risk
        recommendations.append(
            ("🌟", "**Loyalty Reward Acknowledgment:** Send an anniversary appreciation perk or exclusive customer rewards points.")
        )
        recommendations.append(
            ("🤝", "**Advocacy & Referral Invitation:** Customer is highly satisfied; invite them to the ConnectTel customer referral program with bill credits for both parties.")
        )
        recommendations.append(
            ("🔍", "**Routine Service Pulse Check:** Continue standard monitoring with semi-annual satisfaction surveys.")
        )

    # Render recommendation cards
    for icon, text in recommendations:
        st.markdown(
            f"""
            <div class="rec-item">
                <div class="rec-icon">{icon}</div>
                <div class="rec-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Business Interpretation Note
    st.info(
        f"**Model Context:** According to ConnectTel's serialized LightGBM production pipeline, this customer has an estimated **{prob_pct:.2f}%** churn probability. The classification follows calibrated decision thresholds established during model validation on holdout test data."
    )
