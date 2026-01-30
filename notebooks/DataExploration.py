"""
Task 1: Data Exploration and Enrichment
Ethiopia Financial Inclusion Forecasting Project

This script loads, explores, and enriches the financial inclusion dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataEnrichmentPipeline:
    """Pipeline for exploring and enriching Ethiopia FI data"""
    
    def __init__(self, data_path=None, impact_links_path=None, reference_codes_path=None):
        """Initialize with dataset paths"""
        self.data_path = data_path
        self.impact_links_path = impact_links_path
        self.reference_codes_path = reference_codes_path
        self.data = None
        self.impact_links = None
        self.reference_codes = None
        self.enrichment_log = []
        
    def load_data(self):
        """Load all datasets"""
        print("="*80)
        print("LOADING DATASETS")
        print("="*80)
        
        if self.data_path:
            self.data = pd.read_csv(self.data_path)
            print(f"✓ Loaded main dataset: {len(self.data)} records")
            print(f"  Columns: {len(self.data.columns)}")
        else:
            print("⚠ No data path provided - creating sample structure")
            self.create_sample_data()
            
        if self.impact_links_path:
            self.impact_links = pd.read_csv(self.impact_links_path)
            print(f"✓ Loaded impact links: {len(self.impact_links)} records")
        else:
            print("⚠ No impact links path provided")
            
        if self.reference_codes_path:
            self.reference_codes = pd.read_csv(self.reference_codes_path)
            print(f"✓ Loaded reference codes: {len(self.reference_codes)} records")
        else:
            print("⚠ No reference codes path provided")
            
        print()
        
    def create_sample_data(self):
        """Create sample data structure based on schema"""
        columns = [
            'record_id', 'parent_id', 'record_type', 'category', 'pillar',
            'indicator', 'indicator_code', 'indicator_direction', 'value_numeric',
            'value_text', 'value_type', 'unit', 'observation_date', 'period_start',
            'period_end', 'fiscal_year', 'gender', 'location', 'region',
            'source_name', 'source_type', 'source_url', 'confidence',
            'related_indicator', 'relationship_type', 'impact_direction',
            'impact_magnitude', 'impact_estimate', 'lag_months', 'evidence_basis',
            'comparable_country', 'collected_by', 'collection_date', 'original_text', 'notes'
        ]
        self.data = pd.DataFrame(columns=columns)
        
    def explore_schema(self):
        """Understand the data schema and structure"""
        print("="*80)
        print("1. SCHEMA EXPLORATION")
        print("="*80)
        
        if self.data is None or len(self.data) == 0:
            print("⚠ No data loaded to explore")
            return
            
        print(f"\nDataset Shape: {self.data.shape}")
        print(f"Columns ({len(self.data.columns)}): {', '.join(self.data.columns[:10])}...")
        
        print("\nColumn Data Types:")
        print(self.data.dtypes.value_counts())
        
        print("\nMemory Usage:")
        print(f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print()
        
    def explore_record_types(self):
        """Analyze distribution of record types"""
        print("="*80)
        print("2. RECORD TYPE EXPLORATION")
        print("="*80)
        
        if 'record_type' not in self.data.columns:
            print("⚠ 'record_type' column not found")
            return
            
        print("\nRecord Type Distribution:")
        record_counts = self.data['record_type'].value_counts()
        for record_type, count in record_counts.items():
            pct = (count / len(self.data)) * 100
            print(f"  {record_type:15s}: {count:4d} ({pct:5.1f}%)")
            
        # Analyze by pillar
        if 'pillar' in self.data.columns:
            print("\nRecords by Pillar:")
            pillar_counts = self.data['pillar'].value_counts(dropna=False)
            for pillar, count in pillar_counts.items():
                print(f"  {str(pillar):15s}: {count:4d}")
                
        # Analyze by source type
        if 'source_type' in self.data.columns:
            print("\nRecords by Source Type:")
            source_counts = self.data['source_type'].value_counts(dropna=False)
            for source, count in source_counts.items():
                print(f"  {str(source):15s}: {count:4d}")
                
        # Analyze by confidence
        if 'confidence' in self.data.columns:
            print("\nRecords by Confidence Level:")
            conf_counts = self.data['confidence'].value_counts(dropna=False)
            for conf, count in conf_counts.items():
                print(f"  {str(conf):15s}: {count:4d}")
                
        print()
        
    def explore_temporal_coverage(self):
        """Analyze temporal range of observations"""
        print("="*80)
        print("3. TEMPORAL COVERAGE")
        print("="*80)
        
        if 'observation_date' not in self.data.columns:
            print("⚠ 'observation_date' column not found")
            return
            
        # Convert to datetime
        self.data['observation_date'] = pd.to_datetime(self.data['observation_date'], errors='coerce')
        
        observations = self.data[self.data['record_type'] == 'observation'].copy()
        
        if len(observations) == 0:
            print("⚠ No observation records found")
            return
            
        valid_dates = observations['observation_date'].dropna()
        
        if len(valid_dates) > 0:
            print(f"\nTemporal Range:")
            print(f"  Earliest: {valid_dates.min().strftime('%Y-%m-%d')}")
            print(f"  Latest:   {valid_dates.max().strftime('%Y-%m-%d')}")
            print(f"  Span:     {(valid_dates.max() - valid_dates.min()).days} days")
            
            print(f"\nObservations by Year:")
            year_counts = valid_dates.dt.year.value_counts().sort_index()
            for year, count in year_counts.items():
                print(f"  {year}: {count:4d} observations")
        else:
            print("⚠ No valid dates found")
            
        print()
        
    def explore_indicators(self):
        """List all unique indicators and their coverage"""
        print("="*80)
        print("4. INDICATOR EXPLORATION")
        print("="*80)
        
        if 'indicator_code' not in self.data.columns:
            print("⚠ 'indicator_code' column not found")
            return
            
        observations = self.data[self.data['record_type'] == 'observation'].copy()
        
        print(f"\nUnique Indicators: {observations['indicator_code'].nunique()}")
        
        indicator_summary = observations.groupby('indicator_code').agg({
            'value_numeric': 'count',
            'observation_date': lambda x: pd.to_datetime(x).min().strftime('%Y') if pd.to_datetime(x).notna().any() else 'N/A'
        }).rename(columns={'value_numeric': 'Count', 'observation_date': 'Earliest'})
        
        print("\nIndicator Coverage:")
        for idx, row in indicator_summary.iterrows():
            print(f"  {idx:25s}: {int(row['Count']):3d} observations (from {row['Earliest']})")
            
        print()
        
    def explore_events(self):
        """Understand which events are cataloged"""
        print("="*80)
        print("5. EVENT CATALOG")
        print("="*80)
        
        events = self.data[self.data['record_type'] == 'event'].copy()
        
        if len(events) == 0:
            print("⚠ No events found in dataset")
            return
            
        print(f"\nTotal Events: {len(events)}")
        
        if 'category' in events.columns:
            print("\nEvents by Category:")
            cat_counts = events['category'].value_counts()
            for cat, count in cat_counts.items():
                print(f"  {cat:20s}: {count:3d}")
                
        if 'observation_date' in events.columns:
            events['observation_date'] = pd.to_datetime(events['observation_date'], errors='coerce')
            valid_events = events.dropna(subset=['observation_date']).sort_values('observation_date')
            
            print("\nEvent Timeline:")
            for _, event in valid_events.iterrows():
                date = event['observation_date'].strftime('%Y-%m-%d')
                indicator = event.get('indicator', 'N/A')
                category = event.get('category', 'N/A')
                print(f"  {date}: [{category:15s}] {indicator}")
                
        print()
        
    def explore_impact_links(self):
        """Review existing impact links and relationships"""
        print("="*80)
        print("6. IMPACT LINKS EXPLORATION")
        print("="*80)
        
        impact_links = self.data[self.data['record_type'] == 'impact_link'].copy()
        
        if len(impact_links) == 0:
            print("⚠ No impact_link records found")
            return
            
        print(f"\nTotal Impact Links: {len(impact_links)}")
        
        if 'impact_direction' in impact_links.columns:
            print("\nBy Impact Direction:")
            dir_counts = impact_links['impact_direction'].value_counts()
            for direction, count in dir_counts.items():
                print(f"  {direction:15s}: {count:3d}")
                
        if 'impact_magnitude' in impact_links.columns:
            print("\nBy Impact Magnitude:")
            mag_counts = impact_links['impact_magnitude'].value_counts()
            for magnitude, count in mag_counts.items():
                print(f"  {magnitude:15s}: {count:3d}")
                
        if 'related_indicator' in impact_links.columns:
            print("\nBy Related Indicator:")
            ind_counts = impact_links['related_indicator'].value_counts()
            for indicator, count in ind_counts.items():
                print(f"  {indicator:25s}: {count:3d}")
                
        print("\nSample Impact Links:")
        display_cols = ['parent_id', 'related_indicator', 'impact_direction', 
                       'impact_magnitude', 'lag_months']
        available_cols = [col for col in display_cols if col in impact_links.columns]
        print(impact_links[available_cols].head(10).to_string(index=False))
        
        print()
        
    def add_observation(self, record_id, pillar, indicator, indicator_code, 
                       value_numeric, observation_date, source_name, source_url,
                       confidence, original_text, notes, **kwargs):
        """Add a new observation record"""
        new_record = {
            'record_id': record_id,
            'record_type': 'observation',
            'pillar': pillar,
            'indicator': indicator,
            'indicator_code': indicator_code,
            'value_numeric': value_numeric,
            'observation_date': observation_date,
            'source_name': source_name,
            'source_url': source_url,
            'confidence': confidence,
            'original_text': original_text,
            'notes': notes,
            'collected_by': 'Python Script',
            'collection_date': datetime.now().strftime('%Y-%m-%d'),
            **kwargs
        }
        
        self.data = pd.concat([self.data, pd.DataFrame([new_record])], ignore_index=True)
        
        self.enrichment_log.append({
            'record_id': record_id,
            'record_type': 'observation',
            'action': 'added',
            'reason': notes
        })
        
        print(f"✓ Added observation: {record_id} - {indicator}")
        
    def add_event(self, record_id, category, indicator, observation_date,
                 source_name, source_url, confidence, original_text, notes, **kwargs):
        """Add a new event record"""
        new_record = {
            'record_id': record_id,
            'record_type': 'event',
            'category': category,
            'indicator': indicator,
            'observation_date': observation_date,
            'source_name': source_name,
            'source_url': source_url,
            'confidence': confidence,
            'original_text': original_text,
            'notes': notes,
            'collected_by': 'Python Script',
            'collection_date': datetime.now().strftime('%Y-%m-%d'),
            **kwargs
        }
        
        self.data = pd.concat([self.data, pd.DataFrame([new_record])], ignore_index=True)
        
        self.enrichment_log.append({
            'record_id': record_id,
            'record_type': 'event',
            'action': 'added',
            'reason': notes
        })
        
        print(f"✓ Added event: {record_id} - {indicator}")
        
    def add_impact_link(self, record_id, parent_id, pillar, related_indicator,
                       impact_direction, impact_magnitude, lag_months,
                       evidence_basis, notes, **kwargs):
        """Add a new impact_link record"""
        new_record = {
            'record_id': record_id,
            'parent_id': parent_id,
            'record_type': 'impact_link',
            'pillar': pillar,
            'related_indicator': related_indicator,
            'impact_direction': impact_direction,
            'impact_magnitude': impact_magnitude,
            'lag_months': lag_months,
            'evidence_basis': evidence_basis,
            'notes': notes,
            'collected_by': 'Python Script',
            'collection_date': datetime.now().strftime('%Y-%m-%d'),
            **kwargs
        }
        
        self.data = pd.concat([self.data, pd.DataFrame([new_record])], ignore_index=True)
        
        self.enrichment_log.append({
            'record_id': record_id,
            'record_type': 'impact_link',
            'action': 'added',
            'reason': notes
        })
        
        print(f"✓ Added impact link: {record_id} - {parent_id} → {related_indicator}")
        
    def enrich_with_example_data(self):
        """Add example enrichment data (template for actual enrichment)"""
        print("="*80)
        print("7. DATA ENRICHMENT")
        print("="*80)
        print("\nAdding example enrichment records...")
        print()
        
        # Example: Gender-disaggregated account ownership
        self.add_observation(
            record_id='OBS_NEW_001',
            pillar='ACCESS',
            indicator='Account Ownership - Male',
            indicator_code='ACC_OWN_MALE',
            value_numeric=56.0,
            observation_date='2021-12-31',
            source_name='World Bank Global Findex',
            source_url='https://globalfindex.worldbank.org/',
            confidence='high',
            gender='male',
            location='national',
            unit='%',
            value_type='percentage',
            original_text='56% of Ethiopian males had account in 2021',
            notes='Gender-disaggregated data for ACCESS pillar analysis'
        )
        
        self.add_observation(
            record_id='OBS_NEW_002',
            pillar='ACCESS',
            indicator='Account Ownership - Female',
            indicator_code='ACC_OWN_FEMALE',
            value_numeric=36.0,
            observation_date='2021-12-31',
            source_name='World Bank Global Findex',
            source_url='https://globalfindex.worldbank.org/',
            confidence='high',
            gender='female',
            location='national',
            unit='%',
            value_type='percentage',
            original_text='36% of Ethiopian females had account in 2021',
            notes='Gender-disaggregated data showing 20pp gender gap'
        )
        
        # Example: Infrastructure data
        self.add_observation(
            record_id='OBS_NEW_003',
            pillar='ACCESS',
            indicator='4G Coverage',
            indicator_code='INFRA_4G_COV',
            value_numeric=65.0,
            observation_date='2023-12-31',
            source_name='GSMA Mobile Connectivity Index',
            source_url='https://www.mobileconnectivityindex.com/',
            confidence='medium',
            location='national',
            unit='%',
            value_type='percentage',
            original_text='4G coverage reached 65% of population in 2023',
            notes='Infrastructure enabler for digital financial services'
        )
        
        # Example: New event
        self.add_event(
            record_id='EVT_NEW_001',
            category='regulation',
            indicator='Payment Interoperability Launch',
            observation_date='2022-06-01',
            source_name='National Bank of Ethiopia',
            source_url='https://nbe.gov.et/',
            confidence='high',
            original_text='NBE launched interoperable payment system in June 2022',
            notes='Key infrastructure enabling P2P transfers across platforms'
        )
        
        # Example: New impact link
        self.add_impact_link(
            record_id='IMP_NEW_001',
            parent_id='EVT_NEW_001',
            pillar='USAGE',
            related_indicator='DIGITAL_PAYMENT_RATE',
            impact_direction='increase',
            impact_magnitude='medium',
            lag_months=6,
            evidence_basis='empirical',
            notes='Interoperability expected to boost digital payment adoption'
        )
        
        print(f"\n✓ Added {len(self.enrichment_log)} enrichment records")
        print()
        
    def generate_enrichment_log(self):
        """Generate markdown documentation of enrichment activities"""
        print("="*80)
        print("8. GENERATING ENRICHMENT LOG")
        print("="*80)
        
        log_content = "# Data Enrichment Log\n\n"
        log_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        log_content += f"**Total Enrichments:** {len(self.enrichment_log)}\n\n"
        
        log_content += "## Summary by Record Type\n\n"
        if self.enrichment_log:
            log_df = pd.DataFrame(self.enrichment_log)
            summary = log_df['record_type'].value_counts()
            for record_type, count in summary.items():
                log_content += f"- **{record_type}**: {count} records\n"
        else:
            log_content += "No enrichments performed.\n"
            
        log_content += "\n## Detailed Enrichment Records\n\n"
        for i, entry in enumerate(self.enrichment_log, 1):
            log_content += f"### {i}. {entry['record_id']}\n"
            log_content += f"- **Type:** {entry['record_type']}\n"
            log_content += f"- **Action:** {entry['action']}\n"
            log_content += f"- **Reason:** {entry['reason']}\n\n"
            
        # Save to file
        log_path = '/home/claude/ethiopia_fi_project/data_enrichment_log.md'
        with open(log_path, 'w') as f:
            f.write(log_content)
            
        print(f"✓ Enrichment log saved to: {log_path}")
        print()
        
    def save_enriched_data(self, output_path=None):
        """Save the enriched dataset"""
        if output_path is None:
            output_path = '/home/claude/ethiopia_fi_project/ethiopia_fi_unified_data_enriched.csv'
            
        self.data.to_csv(output_path, index=False)
        print(f"✓ Enriched dataset saved to: {output_path}")
        print(f"  Total records: {len(self.data)}")
        print()
        
    def run_full_pipeline(self):
        """Execute the complete Task 1 pipeline"""
        print("\n" + "="*80)
        print("ETHIOPIA FINANCIAL INCLUSION - TASK 1: DATA ENRICHMENT")
        print("="*80 + "\n")
        
        self.load_data()
        self.explore_schema()
        self.explore_record_types()
        self.explore_temporal_coverage()
        self.explore_indicators()
        self.explore_events()
        self.explore_impact_links()
        self.enrich_with_example_data()
        self.generate_enrichment_log()
        self.save_enriched_data()
        
        print("="*80)
        print("TASK 1 COMPLETE")
        print("="*80)
        print("\nNext Steps:")
        print("  1. Review the enrichment log at: data_enrichment_log.md")
        print("  2. Add more observations, events, and impact links as needed")
        print("  3. Proceed to Task 2: Exploratory Data Analysis")
        print()


def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = DataEnrichmentPipeline(
        data_path=None,  # Replace with actual path: 'ethiopia_fi_unified_data.csv'
        impact_links_path=None,  # Replace with actual path
        reference_codes_path=None  # Replace with actual path
    )
    
    # Run full pipeline
    pipeline.run_full_pipeline()
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()