"""
Task 2: Exploratory Data Analysis
Ethiopia Financial Inclusion Forecasting Project

This script performs comprehensive EDA on the financial inclusion dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FinancialInclusionEDA:
    """Comprehensive EDA for Ethiopia Financial Inclusion data"""
    
    def __init__(self, data_path=None):
        """Initialize with dataset path"""
        self.data_path = data_path
        self.data = None
        self.key_insights = []
        self.figures = []
        
    def load_data(self):
        """Load the enriched dataset"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        
        if self.data_path:
            self.data = pd.read_csv(self.data_path)
            print(f"✓ Loaded dataset: {len(self.data)} records")
        else:
            print("⚠ No data path provided - using sample data")
            self.create_sample_data()
            
        # Convert date columns
        if 'observation_date' in self.data.columns:
            self.data['observation_date'] = pd.to_datetime(self.data['observation_date'], errors='coerce')
            
        print()
        
    def create_sample_data(self):
        """Create sample data for demonstration"""
        # Historical account ownership data
        account_data = {
            'record_id': ['OBS_001', 'OBS_002', 'OBS_003', 'OBS_004', 'OBS_005'],
            'record_type': ['observation'] * 5,
            'pillar': ['ACCESS'] * 5,
            'indicator': ['Account Ownership'] * 5,
            'indicator_code': ['ACC_OWNERSHIP'] * 5,
            'value_numeric': [14, 22, 35, 46, 49],
            'observation_date': pd.to_datetime(['2011-12-31', '2014-12-31', '2017-12-31', 
                                                '2021-12-31', '2024-12-31']),
            'unit': ['%'] * 5,
            'confidence': ['high'] * 5
        }
        self.data = pd.DataFrame(account_data)
        
    def dataset_overview(self):
        """Provide comprehensive dataset summary"""
        print("="*80)
        print("1. DATASET OVERVIEW")
        print("="*80)
        
        print(f"\nDataset Shape: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
        
        if 'record_type' in self.data.columns:
            print("\nRecords by Type:")
            type_summary = self.data['record_type'].value_counts()
            for rtype, count in type_summary.items():
                pct = (count / len(self.data)) * 100
                print(f"  {rtype:15s}: {count:5d} ({pct:5.1f}%)")
                
        if 'pillar' in self.data.columns:
            print("\nRecords by Pillar:")
            pillar_summary = self.data['pillar'].value_counts(dropna=False)
            for pillar, count in pillar_summary.items():
                pct = (count / len(self.data[self.data['pillar'].notna()])) * 100
                print(f"  {str(pillar):15s}: {count:5d} ({pct:5.1f}%)")
                
        if 'source_type' in self.data.columns:
            print("\nRecords by Source Type:")
            source_summary = self.data['source_type'].value_counts(dropna=False)
            for source, count in source_summary.items():
                print(f"  {str(source):20s}: {count:5d}")
                
        print()
        
    def temporal_coverage_analysis(self):
        """Visualize temporal coverage of indicators"""
        print("="*80)
        print("2. TEMPORAL COVERAGE ANALYSIS")
        print("="*80)
        
        observations = self.data[self.data['record_type'] == 'observation'].copy()
        
        if len(observations) == 0:
            print("⚠ No observation records found")
            return
            
        valid_obs = observations[observations['observation_date'].notna()].copy()
        valid_obs['year'] = valid_obs['observation_date'].dt.year
        
        # Coverage by year and indicator
        if 'indicator_code' in valid_obs.columns:
            coverage_matrix = valid_obs.pivot_table(
                index='indicator_code',
                columns='year',
                values='value_numeric',
                aggfunc='count',
                fill_value=0
            )
            
            print(f"\nIndicators with temporal data: {len(coverage_matrix)}")
            print(f"Years covered: {coverage_matrix.columns.min()} - {coverage_matrix.columns.max()}")
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(coverage_matrix, annot=True, fmt='g', cmap='YlGnBu', 
                       cbar_kws={'label': 'Number of Observations'}, ax=ax)
            ax.set_title('Temporal Coverage of Financial Inclusion Indicators', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Indicator', fontsize=12)
            plt.tight_layout()
            
            fig_path = '/home/claude/ethiopia_fi_project/fig_temporal_coverage.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            self.figures.append(fig_path)
            print(f"✓ Saved figure: {fig_path}")
            plt.close()
        
        print()
        
    def data_quality_assessment(self):
        """Assess data quality and identify gaps"""
        print("="*80)
        print("3. DATA QUALITY ASSESSMENT")
        print("="*80)
        
        # Confidence level distribution
        if 'confidence' in self.data.columns:
            print("\nConfidence Level Distribution:")
            conf_dist = self.data['confidence'].value_counts(dropna=False)
            for conf, count in conf_dist.items():
                pct = (count / len(self.data)) * 100
                print(f"  {str(conf):10s}: {count:5d} ({pct:5.1f}%)")
                
        # Missing data analysis
        print("\nMissing Data Analysis:")
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        }).sort_values('Missing Count', ascending=False)
        
        print(missing_df[missing_df['Missing Count'] > 0].head(10).to_string())
        
        # Identify sparse indicators
        if 'indicator_code' in self.data.columns:
            observations = self.data[self.data['record_type'] == 'observation']
            indicator_counts = observations['indicator_code'].value_counts()
            
            print("\nSparse Indicators (≤ 3 observations):")
            sparse = indicator_counts[indicator_counts <= 3]
            for indicator, count in sparse.items():
                print(f"  {indicator:30s}: {count} observations")
                
        self.key_insights.append({
            'category': 'Data Quality',
            'insight': f"Dataset has {len(self.data)} records with confidence levels: " +
                      f"{dict(self.data['confidence'].value_counts())}"
        })
        
        print()
        
    def analyze_account_ownership(self):
        """Analyze account ownership trajectory (ACCESS pillar)"""
        print("="*80)
        print("4. ACCESS ANALYSIS: ACCOUNT OWNERSHIP")
        print("="*80)
        
        # Filter for account ownership data
        access_data = self.data[
            (self.data['pillar'] == 'ACCESS') & 
            (self.data['indicator_code'].str.contains('ACC_OWN', na=False))
        ].copy()
        
        if len(access_data) == 0:
            print("⚠ No account ownership data found")
            return
            
        # Sort by date
        access_data = access_data.sort_values('observation_date')
        
        # Calculate growth rates
        access_data['year'] = access_data['observation_date'].dt.year
        
        print("\nHistorical Account Ownership:")
        for _, row in access_data.iterrows():
            year = row['observation_date'].year if pd.notna(row['observation_date']) else 'N/A'
            value = row['value_numeric'] if pd.notna(row['value_numeric']) else 'N/A'
            print(f"  {year}: {value}%")
            
        # Calculate period growth
        if len(access_data) > 1:
            print("\nGrowth Between Surveys:")
            for i in range(1, len(access_data)):
                prev_val = access_data.iloc[i-1]['value_numeric']
                curr_val = access_data.iloc[i]['value_numeric']
                prev_year = access_data.iloc[i-1]['observation_date'].year
                curr_year = access_data.iloc[i]['observation_date'].year
                
                if pd.notna(prev_val) and pd.notna(curr_val):
                    growth = curr_val - prev_val
                    years = curr_year - prev_year
                    annual = growth / years if years > 0 else 0
                    print(f"  {prev_year}-{curr_year}: +{growth:.1f}pp over {years} years " +
                          f"(~{annual:.1f}pp/year)")
                          
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        years = access_data['observation_date'].dt.year
        values = access_data['value_numeric']
        
        ax.plot(years, values, marker='o', linewidth=2, markersize=10, 
               color='#2E75B5', label='Account Ownership')
        
        # Add annotations
        for year, value in zip(years, values):
            ax.annotate(f'{value:.0f}%', 
                       xy=(year, value), 
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=10,
                       fontweight='bold')
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Account Ownership (%)', fontsize=12, fontweight='bold')
        ax.set_title('Ethiopia: Financial Account Ownership Trend (2011-2024)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        fig_path = '/home/claude/ethiopia_fi_project/fig_account_ownership_trend.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        self.figures.append(fig_path)
        print(f"\n✓ Saved figure: {fig_path}")
        plt.close()
        
        # Key insight about slowdown
        if len(access_data) >= 2:
            latest_growth = access_data.iloc[-1]['value_numeric'] - access_data.iloc[-2]['value_numeric']
            self.key_insights.append({
                'category': 'Access',
                'insight': f"Account ownership grew only +{latest_growth:.0f}pp in the most recent period, " +
                          "indicating significant slowdown despite mobile money expansion"
            })
        
        print()
        
    def analyze_gender_gap(self):
        """Analyze gender disparities in financial inclusion"""
        print("="*80)
        print("5. GENDER GAP ANALYSIS")
        print("="*80)
        
        gender_data = self.data[
            (self.data['gender'].isin(['male', 'female'])) &
            (self.data['pillar'] == 'ACCESS')
        ].copy()
        
        if len(gender_data) == 0:
            print("⚠ No gender-disaggregated data available")
            print("   Recommendation: Enrich dataset with Findex gender breakdowns")
            return
            
        # Calculate gender gap
        pivot = gender_data.pivot_table(
            index='observation_date',
            columns='gender',
            values='value_numeric'
        )
        
        if 'male' in pivot.columns and 'female' in pivot.columns:
            pivot['gap'] = pivot['male'] - pivot['female']
            
            print("\nGender Gap in Account Ownership:")
            for date, row in pivot.iterrows():
                year = date.year if pd.notna(date) else 'N/A'
                male = row['male'] if pd.notna(row['male']) else 'N/A'
                female = row['female'] if pd.notna(row['female']) else 'N/A'
                gap = row['gap'] if pd.notna(row['gap']) else 'N/A'
                print(f"  {year}: Male {male}%, Female {female}%, Gap {gap}pp")
                
            self.key_insights.append({
                'category': 'Gender',
                'insight': f"Significant gender gap of {pivot['gap'].iloc[-1]:.0f}pp in account ownership, " +
                          "with males substantially ahead of females"
            })
        
        print()
        
    def analyze_digital_payments(self):
        """Analyze digital payment adoption (USAGE pillar)"""
        print("="*80)
        print("6. USAGE ANALYSIS: DIGITAL PAYMENTS")
        print("="*80)
        
        usage_data = self.data[
            (self.data['pillar'] == 'USAGE') |
            (self.data['indicator_code'].str.contains('DIGITAL|MOBILE|PAYMENT', na=False))
        ].copy()
        
        if len(usage_data) == 0:
            print("⚠ Limited digital payment data available")
            return
            
        usage_data = usage_data.sort_values('observation_date')
        
        print("\nDigital Payment Indicators:")
        for _, row in usage_data.iterrows():
            indicator = row.get('indicator', 'N/A')
            value = row.get('value_numeric', 'N/A')
            year = row['observation_date'].year if pd.notna(row.get('observation_date')) else 'N/A'
            print(f"  {year}: {indicator} = {value}")
            
        self.key_insights.append({
            'category': 'Usage',
            'insight': "Digital payment adoption remains lower than account ownership, " +
                      "indicating a usage gap that needs addressing"
        })
        
        print()
        
    def analyze_infrastructure(self):
        """Analyze infrastructure enablers"""
        print("="*80)
        print("7. INFRASTRUCTURE ANALYSIS")
        print("="*80)
        
        infra_data = self.data[
            (self.data['indicator_code'].str.contains('INFRA|4G|MOBILE|ATM|AGENT', na=False)) |
            (self.data['category'] == 'infrastructure')
        ].copy()
        
        if len(infra_data) == 0:
            print("⚠ Limited infrastructure data available")
            print("   Recommendation: Add data on 4G coverage, mobile penetration, agent density")
            return
            
        print("\nInfrastructure Indicators:")
        for _, row in infra_data.iterrows():
            indicator = row.get('indicator', 'N/A')
            value = row.get('value_numeric', 'N/A')
            year = row['observation_date'].year if pd.notna(row.get('observation_date')) else 'N/A'
            print(f"  {year}: {indicator} = {value}")
            
        print()
        
    def analyze_event_timeline(self):
        """Create event timeline and analyze relationships with indicators"""
        print("="*80)
        print("8. EVENT TIMELINE ANALYSIS")
        print("="*80)
        
        events = self.data[self.data['record_type'] == 'event'].copy()
        
        if len(events) == 0:
            print("⚠ No events found in dataset")
            return
            
        events = events.sort_values('observation_date')
        
        print(f"\nCataloged Events: {len(events)}")
        print("\nEvent Timeline:")
        for _, event in events.iterrows():
            date = event['observation_date'].strftime('%Y-%m-%d') if pd.notna(event['observation_date']) else 'N/A'
            indicator = event.get('indicator', 'N/A')
            category = event.get('category', 'N/A')
            print(f"  {date}: [{category:15s}] {indicator}")
            
        # Create timeline visualization
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot events
        event_dates = pd.to_datetime(events['observation_date'].dropna())
        event_names = events[events['observation_date'].notna()]['indicator'].values
        
        for i, (date, name) in enumerate(zip(event_dates, event_names)):
            ax.axvline(x=date, color='red', alpha=0.5, linestyle='--', linewidth=1)
            ax.text(date, 0.9 - (i % 5) * 0.15, name, rotation=45, ha='right', 
                   fontsize=8, color='red')
        
        # Overlay account ownership trend if available
        access_data = self.data[
            (self.data['pillar'] == 'ACCESS') & 
            (self.data['indicator_code'].str.contains('ACC_OWN', na=False))
        ].sort_values('observation_date')
        
        if len(access_data) > 0:
            ax2 = ax.twinx()
            ax2.plot(access_data['observation_date'], access_data['value_numeric'],
                    marker='o', linewidth=2, markersize=8, color='#2E75B5',
                    label='Account Ownership')
            ax2.set_ylabel('Account Ownership (%)', fontsize=11, fontweight='bold')
            ax2.legend(loc='upper left')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_title('Event Timeline: Impact on Financial Inclusion', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        plt.tight_layout()
        
        fig_path = '/home/claude/ethiopia_fi_project/fig_event_timeline.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        self.figures.append(fig_path)
        print(f"\n✓ Saved figure: {fig_path}")
        plt.close()
        
        self.key_insights.append({
            'category': 'Events',
            'insight': f"{len(events)} key events identified that may impact financial inclusion outcomes"
        })
        
        print()
        
    def correlation_analysis(self):
        """Analyze correlations between indicators"""
        print("="*80)
        print("9. CORRELATION ANALYSIS")
        print("="*80)
        
        observations = self.data[self.data['record_type'] == 'observation'].copy()
        
        # Create pivot table of indicators over time
        if 'indicator_code' in observations.columns and len(observations) > 0:
            pivot = observations.pivot_table(
                index='observation_date',
                columns='indicator_code',
                values='value_numeric'
            )
            
            # Calculate correlations
            corr_matrix = pivot.corr()
            
            if len(corr_matrix) > 1:
                print("\nTop Correlations with Account Ownership:")
                if 'ACC_OWNERSHIP' in corr_matrix.columns:
                    acc_corr = corr_matrix['ACC_OWNERSHIP'].sort_values(ascending=False)
                    for indicator, corr in acc_corr.items():
                        if indicator != 'ACC_OWNERSHIP' and pd.notna(corr):
                            print(f"  {indicator:30s}: {corr:6.3f}")
                            
                # Create correlation heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                           center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
                ax.set_title('Indicator Correlation Matrix', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                fig_path = '/home/claude/ethiopia_fi_project/fig_correlation_matrix.png'
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                self.figures.append(fig_path)
                print(f"\n✓ Saved figure: {fig_path}")
                plt.close()
        else:
            print("⚠ Insufficient data for correlation analysis")
            
        print()
        
    def summarize_key_insights(self):
        """Generate summary of key insights"""
        print("="*80)
        print("10. KEY INSIGHTS SUMMARY")
        print("="*80)
        
        print(f"\nTotal Insights Generated: {len(self.key_insights)}\n")
        
        for i, insight in enumerate(self.key_insights, 1):
            print(f"{i}. [{insight['category']}] {insight['insight']}")
            
        # Add general insights based on project context
        general_insights = [
            {
                'category': 'Slowdown',
                'insight': "The dramatic slowdown in account ownership growth (+3pp 2021-2024 vs +11pp 2017-2021) " +
                          "despite 65M+ mobile money accounts suggests registration ≠ active usage"
            },
            {
                'category': 'Market Context',
                'insight': "P2P payments dominate Ethiopia's digital finance landscape, used for commerce " +
                          "rather than just transfers, indicating unique market dynamics"
            },
            {
                'category': 'Forecasting',
                'insight': "Limited historical data (only 5 Findex surveys) poses challenges for forecasting; " +
                          "need to leverage high-frequency indicators and event-based modeling"
            }
        ]
        
        for insight in general_insights:
            if insight not in self.key_insights:
                self.key_insights.append(insight)
                
        print()
        
    def document_limitations(self):
        """Document data limitations and gaps"""
        print("="*80)
        print("11. DATA LIMITATIONS & GAPS")
        print("="*80)
        
        limitations = [
            "Historical Data: Only 5 Findex surveys (2011, 2014, 2017, 2021, 2024) provide baseline measurements",
            "Geographic Coverage: Limited regional and urban/rural disaggregation",
            "Gender Data: Need more comprehensive gender-disaggregated indicators",
            "High-Frequency Data: Lack of monthly/quarterly indicators for interim forecasting",
            "Usage Metrics: Limited data on active vs registered users and transaction patterns",
            "Infrastructure: Need comprehensive data on agent networks, POS terminals, 4G coverage",
            "Policy Impact: Limited documentation of regulatory changes and their timing"
        ]
        
        print("\nIdentified Limitations:")
        for i, limitation in enumerate(limitations, 1):
            print(f"  {i}. {limitation}")
            
        print("\nRecommendations for Next Phase:")
        print("  • Enrich with GSMA Mobile Money data for monthly active users")
        print("  • Add ITU telecommunications data for infrastructure coverage")
        print("  • Include NBE quarterly reports for banking sector metrics")
        print("  • Incorporate operator-specific data from Telebirr and M-Pesa")
        print("  • Model event impacts using comparable country experiences (Kenya, Tanzania)")
        
        print()
        
    def generate_eda_report(self):
        """Generate markdown report of EDA findings"""
        print("="*80)
        print("12. GENERATING EDA REPORT")
        print("="*80)
        
        report = "# Exploratory Data Analysis Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "---\n\n"
        
        report += "## Executive Summary\n\n"
        report += f"This analysis examined **{len(self.data)} records** from Ethiopia's financial inclusion dataset. "
        report += f"A total of **{len(self.key_insights)} key insights** were identified across Access, Usage, Gender, "
        report += "Infrastructure, and Event dimensions.\n\n"
        
        report += "## Key Findings\n\n"
        for i, insight in enumerate(self.key_insights, 1):
            report += f"{i}. **[{insight['category']}]** {insight['insight']}\n"
            
        report += "\n## Data Quality Assessment\n\n"
        report += f"- **Total Records:** {len(self.data)}\n"
        if 'record_type' in self.data.columns:
            for rtype, count in self.data['record_type'].value_counts().items():
                report += f"- **{rtype.title()}:** {count}\n"
                
        report += "\n## Visualizations Generated\n\n"
        for fig_path in self.figures:
            fig_name = fig_path.split('/')[-1]
            report += f"- `{fig_name}`\n"
            
        report += "\n## Next Steps\n\n"
        report += "1. **Impact Modeling:** Quantify effects of Telebirr launch, M-Pesa entry, and policy changes\n"
        report += "2. **Forecasting:** Develop models for 2025-2027 projections\n"
        report += "3. **Scenario Analysis:** Model different growth trajectories under various assumptions\n"
        report += "4. **Dashboard Development:** Create interactive visualization for stakeholders\n"
        
        # Save report
        report_path = '/home/claude/ethiopia_fi_project/eda_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
            
        print(f"✓ EDA report saved to: {report_path}")
        print()
        
    def run_full_analysis(self):
        """Execute the complete Task 2 pipeline"""
        print("\n" + "="*80)
        print("ETHIOPIA FINANCIAL INCLUSION - TASK 2: EXPLORATORY DATA ANALYSIS")
        print("="*80 + "\n")
        
        self.load_data()
        self.dataset_overview()
        self.temporal_coverage_analysis()
        self.data_quality_assessment()
        self.analyze_account_ownership()
        self.analyze_gender_gap()
        self.analyze_digital_payments()
        self.analyze_infrastructure()
        self.analyze_event_timeline()
        self.correlation_analysis()
        self.summarize_key_insights()
        self.document_limitations()
        self.generate_eda_report()
        
        print("="*80)
        print("TASK 2 COMPLETE")
        print("="*80)
        print(f"\nGenerated {len(self.figures)} visualizations")
        print(f"Identified {len(self.key_insights)} key insights")
        print("\nOutputs:")
        print("  • EDA Report: eda_report.md")
        print("  • Visualizations: fig_*.png")
        print("\nReady to proceed to forecasting phase!")
        print()


def main():
    """Main execution function"""
    # Initialize EDA
    eda = FinancialInclusionEDA(
        data_path='/home/claude/ethiopia_fi_project/ethiopia_fi_unified_data_enriched.csv'
    )
    
    # Run full analysis
    eda.run_full_analysis()
    
    return eda


if __name__ == "__main__":
    eda = main()