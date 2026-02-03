"""
Task 5: Interactive Dashboard
Ethiopia Financial Inclusion Forecasting Project

Streamlit dashboard for exploring data, event impacts, and forecasts.

To run: streamlit run task_5_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Ethiopia Financial Inclusion Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E78;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1F4E78;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E75B5;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all project data"""
    base_path = '/home/claude/ethiopia_fi_project'
    
    data = {}
    
    # Main dataset
    main_file = f'{base_path}/ethiopia_fi_unified_data_enriched.csv'
    if os.path.exists(main_file):
        data['main'] = pd.read_csv(main_file)
        data['main']['observation_date'] = pd.to_datetime(
            data['main']['observation_date'], errors='coerce'
        )
    
    # Forecasts
    forecast_file = f'{base_path}/forecasts_2025_2027.csv'
    if os.path.exists(forecast_file):
        data['forecasts'] = pd.read_csv(forecast_file)
    
    # If files don't exist, create sample data
    if not data:
        data = create_sample_data()
    
    return data


def create_sample_data():
    """Create sample data for demonstration"""
    # Historical observations
    obs = pd.DataFrame([
        {'record_type': 'observation', 'indicator_code': 'ACC_OWNERSHIP',
         'value_numeric': 14, 'observation_date': '2011-12-31', 'pillar': 'ACCESS'},
        {'record_type': 'observation', 'indicator_code': 'ACC_OWNERSHIP',
         'value_numeric': 22, 'observation_date': '2014-12-31', 'pillar': 'ACCESS'},
        {'record_type': 'observation', 'indicator_code': 'ACC_OWNERSHIP',
         'value_numeric': 35, 'observation_date': '2017-12-31', 'pillar': 'ACCESS'},
        {'record_type': 'observation', 'indicator_code': 'ACC_OWNERSHIP',
         'value_numeric': 46, 'observation_date': '2021-12-31', 'pillar': 'ACCESS'},
        {'record_type': 'observation', 'indicator_code': 'ACC_OWNERSHIP',
         'value_numeric': 49, 'observation_date': '2024-12-31', 'pillar': 'ACCESS'},
    ])
    obs['observation_date'] = pd.to_datetime(obs['observation_date'])
    
    # Forecasts
    forecasts = pd.DataFrame([
        {'indicator_code': 'ACC_OWNERSHIP', 'year': 2025, 'scenario': 'pessimistic', 'forecast': 50.0},
        {'indicator_code': 'ACC_OWNERSHIP', 'year': 2025, 'scenario': 'base', 'forecast': 52.0},
        {'indicator_code': 'ACC_OWNERSHIP', 'year': 2025, 'scenario': 'optimistic', 'forecast': 54.0},
        {'indicator_code': 'ACC_OWNERSHIP', 'year': 2026, 'scenario': 'pessimistic', 'forecast': 51.0},
        {'indicator_code': 'ACC_OWNERSHIP', 'year': 2026, 'scenario': 'base', 'forecast': 54.0},
        {'indicator_code': 'ACC_OWNERSHIP', 'year': 2026, 'scenario': 'optimistic', 'forecast': 57.0},
        {'indicator_code': 'ACC_OWNERSHIP', 'year': 2027, 'scenario': 'pessimistic', 'forecast': 52.0},
        {'indicator_code': 'ACC_OWNERSHIP', 'year': 2027, 'scenario': 'base', 'forecast': 55.0},
        {'indicator_code': 'ACC_OWNERSHIP', 'year': 2027, 'scenario': 'optimistic', 'forecast': 58.0},
    ])
    
    return {'main': obs, 'forecasts': forecasts}


def overview_page(data):
    """Overview page with key metrics"""
    st.markdown('<div class="main-header">Ethiopia Financial Inclusion Dashboard</div>',
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key metrics
    st.subheader("📊 Current State (2024)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Account Ownership",
            value="49%",
            delta="+3pp since 2021",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Mobile Money Users",
            value="65M+",
            delta="Registered accounts",
            delta_color="off"
        )
    
    with col3:
        st.metric(
            label="Digital Payment Usage",
            value="35%",
            delta="+5pp since 2021",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="Gender Gap",
            value="20pp",
            delta="Male 56% vs Female 36%",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Growth rates
    st.subheader("📈 Historical Growth Analysis")
    
    growth_data = pd.DataFrame({
        'Period': ['2011-2014', '2014-2017', '2017-2021', '2021-2024'],
        'Growth (pp)': [8, 13, 11, 3],
        'Annual Growth': [2.7, 4.3, 2.8, 1.0]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            growth_data,
            x='Period',
            y='Growth (pp)',
            title='Account Ownership Growth by Period',
            color='Growth (pp)',
            color_continuous_scale='RdYlGn',
            text='Growth (pp)'
        )
        fig.update_traces(texttemplate='%{text}pp', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # P2P vs ATM indicator
        st.markdown("### 🔄 P2P/ATM Crossover")
        st.success("**Milestone Reached (2023)**: P2P digital transfers surpassed ATM withdrawals")
        
        st.markdown("""
        **Why it matters**:
        - Signals shift to digital-first behavior
        - P2P used for commerce, not just transfers
        - Indicates strong mobile money adoption
        """)
    
    # Key insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🚨 Growth Slowdown Alert</h4>
        <p>Account ownership grew only <b>+3pp (2021-2024)</b> compared to <b>+11pp (2017-2021)</b>.</p>
        <p><b>Challenge</b>: 65M+ mobile money accounts registered, but only 49% survey ownership.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>🎯 Target Status</h4>
        <p>National target: <b>60% account ownership</b></p>
        <p>Current: <b>49%</b> (11pp gap)</p>
        <p><b>Timeline</b>: Challenging to reach by 2027 under current trajectory</p>
        </div>
        """, unsafe_allow_html=True)


def trends_page(data):
    """Trends analysis page"""
    st.markdown('<div class="main-header">Historical Trends</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Get observations
    if 'main' in data:
        obs = data['main'][data['main']['record_type'] == 'observation'].copy()
    else:
        st.warning("No historical data available")
        return
    
    # Date range selector
    st.subheader("📅 Select Date Range")
    
    date_col1, date_col2 = st.columns(2)
    
    min_date = obs['observation_date'].min()
    max_date = obs['observation_date'].max()
    
    with date_col1:
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    
    with date_col2:
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    
    # Filter data
    mask = (obs['observation_date'] >= pd.to_datetime(start_date)) & \
           (obs['observation_date'] <= pd.to_datetime(end_date))
    filtered_obs = obs[mask]
    
    st.markdown("---")
    
    # Main trend chart
    st.subheader("📊 Account Ownership Trajectory")
    
    acc_data = filtered_obs[filtered_obs['indicator_code'] == 'ACC_OWNERSHIP'].copy()
    
    if len(acc_data) > 0:
        acc_data['year'] = acc_data['observation_date'].dt.year
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=acc_data['year'],
            y=acc_data['value_numeric'],
            mode='lines+markers',
            name='Account Ownership',
            line=dict(color='#2E75B5', width=3),
            marker=dict(size=12, symbol='circle'),
            text=acc_data['value_numeric'].round(1).astype(str) + '%',
            textposition='top center',
            textfont=dict(size=11, color='#2E75B5')
        ))
        
        fig.update_layout(
            title='Account Ownership Rate (2011-2024)',
            xaxis_title='Year',
            yaxis_title='Account Ownership (%)',
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No account ownership data in selected range")
    
    # Pillar comparison
    st.markdown("---")
    st.subheader("📊 Indicator Comparison")
    
    # Group by pillar
    pillar_data = filtered_obs.groupby(['pillar', 'observation_date'])['value_numeric'].mean().reset_index()
    
    if len(pillar_data) > 0:
        fig = px.line(
            pillar_data,
            x='observation_date',
            y='value_numeric',
            color='pillar',
            title='Financial Inclusion Indicators Over Time',
            labels={'value_numeric': 'Value (%)', 'observation_date': 'Date', 'pillar': 'Pillar'}
        )
        
        fig.update_layout(height=450, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)


def forecasts_page(data):
    """Forecasts visualization page"""
    st.markdown('<div class="main-header">Forecasts: 2025-2027</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Scenario selector
    st.subheader("🎯 Select Scenario")
    
    scenario = st.radio(
        "Choose forecast scenario:",
        ['base', 'optimistic', 'pessimistic'],
        format_func=lambda x: x.capitalize(),
        horizontal=True
    )
    
    scenario_descriptions = {
        'base': "Modest recovery, expected event impacts materialize",
        'optimistic': "Accelerated growth, strong policy support, infrastructure expansion",
        'pessimistic': "Continued slowdown, limited event impact, economic headwinds"
    }
    
    st.info(f"**{scenario.capitalize()} Scenario**: {scenario_descriptions[scenario]}")
    
    st.markdown("---")
    
    # Get forecast data
    if 'forecasts' not in data:
        st.warning("No forecast data available")
        return
    
    forecasts = data['forecasts']
    historical = data['main'][data['main']['record_type'] == 'observation'].copy()
    historical = historical[historical['indicator_code'] == 'ACC_OWNERSHIP']
    
    # Filter by scenario
    scenario_forecasts = forecasts[forecasts['scenario'] == scenario]
    
    # Create visualization
    st.subheader("📈 Account Ownership Forecast")
    
    fig = go.Figure()
    
    # Historical data
    if len(historical) > 0:
        historical['year'] = historical['observation_date'].dt.year
        fig.add_trace(go.Scatter(
            x=historical['year'],
            y=historical['value_numeric'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#2E75B5', width=3),
            marker=dict(size=12, symbol='circle')
        ))
    
    # Forecast
    if len(scenario_forecasts) > 0:
        fig.add_trace(go.Scatter(
            x=scenario_forecasts['year'],
            y=scenario_forecasts['forecast'],
            mode='lines+markers',
            name=f'{scenario.capitalize()} Forecast',
            line=dict(color='#00B050', width=3, dash='dash'),
            marker=dict(size=12, symbol='diamond')
        ))
        
        # Add confidence interval if available
        if 'ci_lower' in scenario_forecasts.columns:
            fig.add_trace(go.Scatter(
                x=scenario_forecasts['year'],
                y=scenario_forecasts['ci_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=scenario_forecasts['year'],
                y=scenario_forecasts['ci_lower'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(0, 176, 80, 0.2)',
                fill='tonexty',
                name='95% Confidence',
                hoverinfo='skip'
            ))
    
    # Add 60% target line
    fig.add_hline(
        y=60,
        line_dash="dot",
        line_color="red",
        annotation_text="60% Target (NFIS-II)",
        annotation_position="right"
    )
    
    # Add forecast divider
    fig.add_vline(
        x=2024.5,
        line_dash="dot",
        line_color="gray",
        opacity=0.5
    )
    
    fig.update_layout(
        title=f'Account Ownership: Historical & {scenario.capitalize()} Forecast',
        xaxis_title='Year',
        yaxis_title='Account Ownership (%)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast table
    st.markdown("---")
    st.subheader("📊 Detailed Forecast Values")
    
    col1, col2, col3 = st.columns(3)
    
    for idx, year in enumerate([2025, 2026, 2027]):
        year_data = scenario_forecasts[scenario_forecasts['year'] == year]
        
        if len(year_data) > 0:
            value = year_data.iloc[0]['forecast']
            
            with [col1, col2, col3][idx]:
                st.metric(
                    label=f"{year} Forecast",
                    value=f"{value:.1f}%",
                    delta=f"+{value - 49:.1f}pp from 2024"
                )
    
    # Scenario comparison
    st.markdown("---")
    st.subheader("🔄 Compare All Scenarios")
    
    if st.checkbox("Show scenario comparison"):
        all_scenarios = forecasts[forecasts['indicator_code'] == 'ACC_OWNERSHIP']
        
        fig = go.Figure()
        
        # Historical
        if len(historical) > 0:
            fig.add_trace(go.Scatter(
                x=historical['year'],
                y=historical['value_numeric'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='#2E75B5', width=3)
            ))
        
        # Each scenario
        colors = {'pessimistic': '#C00000', 'base': '#FFA500', 'optimistic': '#00B050'}
        for scen in ['pessimistic', 'base', 'optimistic']:
            scen_data = all_scenarios[all_scenarios['scenario'] == scen]
            fig.add_trace(go.Scatter(
                x=scen_data['year'],
                y=scen_data['forecast'],
                mode='lines+markers',
                name=scen.capitalize(),
                line=dict(color=colors[scen], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Account Ownership: All Scenarios',
            xaxis_title='Year',
            yaxis_title='Account Ownership (%)',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def projections_page(data):
    """Financial inclusion projections and target progress"""
    st.markdown('<div class="main-header">Inclusion Projections & Targets</div>',
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Target progress
    st.subheader("🎯 Progress Toward 60% Target")
    
    current = 49
    target = 60
    progress = (current / target) * 100
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Progress bar
        st.progress(progress / 100)
        
        st.markdown(f"""
        **Current**: {current}% | **Target**: {target}% | **Gap**: {target - current}pp
        
        Progress: **{progress:.1f}%** of target achieved
        """)
    
    with col2:
        st.metric(
            label="Years to Target",
            value="3-5 years",
            delta="Base scenario estimate"
        )
    
    st.markdown("---")
    
    # Scenario analysis
    st.subheader("📊 Target Achievement by Scenario")
    
    scenarios_summary = pd.DataFrame({
        'Scenario': ['Pessimistic', 'Base Case', 'Optimistic'],
        '2025': [50, 52, 54],
        '2026': [51, 54, 57],
        '2027': [52, 55, 58],
        'Target Year': ['2030+', '2028-2029', '2027-2028']
    })
    
    st.dataframe(scenarios_summary, use_container_width=True, hide_index=True)
    
    # Key questions answered
    st.markdown("---")
    st.subheader("❓ Answering Key Consortium Questions")
    
    with st.expander("1. What drives financial inclusion in Ethiopia?"):
        st.markdown("""
        **Primary Drivers:**
        - **Mobile money adoption**: Telebirr (54M users), M-Pesa (10M+)
        - **Infrastructure**: 4G coverage, smartphone penetration, agent networks
        - **Policy**: Interoperability enabling, digital payment mandates
        - **Use cases**: P2P for commerce (unique to Ethiopia)
        
        **Barriers:**
        - Registration ≠ usage gap (65M accounts vs 49% ownership)
        - Gender disparity (20pp gap)
        - Urban-rural divide
        - Limited merchant acceptance
        """)
    
    with st.expander("2. How do events affect inclusion outcomes?"):
        st.markdown("""
        **Major Event Impacts:**
        
        - **Telebirr Launch (May 2021)**: +15pp over 2 years on mobile money
        - **M-Pesa Entry (Aug 2023)**: Expected +5pp on account ownership via competition
        - **Interoperability (2022)**: +10pp on digital payment usage over 3-4 years
        - **Infrastructure investments**: +2-3pp enabling effect
        
        **Impact Patterns:**
        - S-curve adoption: slow start, rapid middle, saturation
        - Lag times: 3-12 months depending on event type
        - Additive effects when multiple events occur
        """)
    
    with st.expander("3. What will inclusion look like in 2025-2027?"):
        st.markdown("""
        **Base Case Projections:**
        
        **Account Ownership:**
        - 2025: 52% (+3pp from 2024)
        - 2026: 54% (+5pp)
        - 2027: 55% (+6pp)
        
        **Digital Payment Usage:**
        - 2025: 38% (+3pp from 2024)
        - 2026: 41% (+6pp)
        - 2027: 44% (+9pp)
        
        **Key Milestones:**
        - P2P dominance continues
        - Merchant payments grow to 15-20% of transactions
        - Gender gap narrows to 15pp
        """)
    
    # Download data
    st.markdown("---")
    st.subheader("💾 Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'forecasts' in data:
            csv = data['forecasts'].to_csv(index=False)
            st.download_button(
                label="📥 Download Forecasts (CSV)",
                data=csv,
                file_name="ethiopia_fi_forecasts_2025_2027.csv",
                mime="text/csv"
            )
    
    with col2:
        if 'main' in data:
            csv = data['main'].to_csv(index=False)
            st.download_button(
                label="📥 Download Historical Data (CSV)",
                data=csv,
                file_name="ethiopia_fi_historical_data.csv",
                mime="text/csv"
            )


def main():
    """Main dashboard function"""
    
    # Load data
    data = load_data()
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    
    page = st.sidebar.radio(
        "Go to:",
        ["Overview", "Trends", "Forecasts", "Projections"]
    )
    
    st.sidebar.markdown("---")
    
    # About section
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.markdown("""
    **Ethiopia Financial Inclusion Forecasting Project**
    
    Client: Selam Analytics
    
    Forecasting financial inclusion indicators (2025-2027) using:
    - Historical survey data
    - Event impact modeling
    - Scenario analysis
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Sources:**")
    st.sidebar.markdown("- World Bank Global Findex")
    st.sidebar.markdown("- National Bank of Ethiopia")
    st.sidebar.markdown("- GSMA Mobile Money")
    
    # Route to pages
    if page == "Overview":
        overview_page(data)
    elif page == "Trends":
        trends_page(data)
    elif page == "Forecasts":
        forecasts_page(data)
    elif page == "Projections":
        projections_page(data)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
    Ethiopia Financial Inclusion Dashboard | Selam Analytics | 2025
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
