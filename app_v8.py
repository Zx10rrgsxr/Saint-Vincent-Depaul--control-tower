from pathlib import Path
from io import BytesIO
import math

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='SVDP Control Tower', page_icon='📦', layout='wide')

def safe_num(x, default=0.0):
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return default
        if isinstance(x, str):
            x = x.replace(',', '').replace('$', '').strip()
        return float(x)
    except Exception:
        return default

def load_workbook_tables(file_bytes: bytes):
    xls = pd.ExcelFile(BytesIO(file_bytes))
    tables = {}
    for name in xls.sheet_names:
        raw = pd.read_excel(BytesIO(file_bytes), sheet_name=name, header=None)
        tables[name] = raw
    return tables

def table_from_header_row(raw: pd.DataFrame, header_row_idx: int) -> pd.DataFrame:
    header = raw.iloc[header_row_idx].tolist()
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = header
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    df.columns = [str(c).strip() if pd.notna(c) else '' for c in df.columns]
    return df

def read_known_tables(raw_sheets: dict):
    slotting = table_from_header_row(raw_sheets['Slotting_Model'], 2)
    monthly_flow = table_from_header_row(raw_sheets['Monthly_Flow'], 2)
    clean_tx = table_from_header_row(raw_sheets['Clean_Transactions'], 2)
    warehouse_capacity = table_from_header_row(raw_sheets['Warehouse_Capacity'], 2)
    return slotting, monthly_flow, clean_tx, warehouse_capacity

def clean_capacity(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work[work['Layout Area'].notna()].copy()
    work['Layout Area'] = work['Layout Area'].astype(str).str.strip()
    work = work[~work['Layout Area'].str.lower().str.contains('total|summary|quantified', na=False)].copy()
    quantified = work[work['Layout Area'].isin(['GIK', 'FOOD', 'GARAGE/KITCHEN'])].copy()
    quantified['Before Pallets'] = quantified['Before Pallets'].apply(safe_num)
    quantified['After Pallets'] = quantified['After Pallets'].apply(safe_num)
    quantified['Increase'] = quantified['After Pallets'] - quantified['Before Pallets']
    order = ['GIK', 'FOOD', 'GARAGE/KITCHEN']
    quantified['__order'] = quantified['Layout Area'].apply(lambda x: order.index(x) if x in order else 99)
    quantified = quantified.sort_values('__order').drop(columns='__order')
    return quantified[['Layout Area', 'Before Pallets', 'After Pallets', 'Increase']]

def extract_additions_note(df: pd.DataFrame) -> bool:
    if 'Layout Area' not in df.columns:
        return False
    work = df.copy()
    work = work[work['Layout Area'].notna()].copy()
    work['Layout Area'] = work['Layout Area'].astype(str).str.strip()
    return any(work['Layout Area'].str.upper() == 'ADDITIONS')

def compute_flow_score(freq: str, mapping: dict) -> float:
    return mapping.get(str(freq).strip(), 2.0)

def compute_item_type_score(item_type: str, mapping: dict) -> float:
    return mapping.get(str(item_type).strip(), 3.0)

def infer_preferred_area(item: str, category: str, item_type: str) -> str:
    item = str(item).strip().lower()
    category = str(category).strip().lower()
    item_type = str(item_type).strip().lower()
    if 'water' in item:
        return 'GIK'
    if item in {'peanut butter', 'chicken leg quarters'}:
        return 'GIK'
    if item == 'ground beef':
        return 'ADDITIONS'
    if item in {'tuna', 'rice', 'black beans', 'pasta sauce', 'green beans', 'chili - beef', 'macaroni in tomato sauce'}:
        return 'FOOD'
    if item in {'oats', 'pancake mix', 'peaches', 'mixed vegetables'}:
        return 'GARAGE/KITCHEN'
    if category == 'water':
        return 'GIK'
    if category == 'meat':
        return 'ADDITIONS'
    if item_type == 'bulk':
        return 'GIK'
    return 'FOOD'

def qty_score(v: float) -> int:
    if v >= 1000:
        return 5
    if v >= 500:
        return 4
    if v >= 250:
        return 3
    if v >= 100:
        return 2
    return 1

def build_model(slotting_df: pd.DataFrame, controls: dict, new_items_df: pd.DataFrame) -> pd.DataFrame:
    df = slotting_df.copy()
    df = df[df['Item'].notna()].copy()
    if not new_items_df.empty:
        df = pd.concat([df, new_items_df], ignore_index=True)

    required_cols = ['Item','Category','Item_Type','Usage_Rate','Qty_Received','Qty_Held','Flow_Priority','Inbound_Freq','Outbound_Freq','Current_Location']
    for c in required_cols:
        if c not in df.columns:
            df[c] = None

    for c in ['Usage_Rate', 'Qty_Received', 'Qty_Held', 'Flow_Priority']:
        df[c] = df[c].apply(safe_num)

    peanut = df['Item'].astype(str).str.strip().str.lower().eq('peanut butter')
    df.loc[peanut, 'Item_Type'] = 'Bulk'
    df.loc[peanut, 'Inbound_Freq'] = '4x/Week'
    df.loc[peanut, 'Outbound_Freq'] = '4x/Week'
    df.loc[peanut, 'Flow_Priority'] = df.loc[peanut, 'Flow_Priority'].clip(lower=controls['peanut_flow_priority'])

    fresh = df['Item'].astype(str).str.strip().str.lower().isin({'chicken leg quarters', 'ground beef'})
    df.loc[fresh, 'Outbound_Freq'] = 'Daily'
    df.loc[fresh, 'Flow_Priority'] = df.loc[fresh, 'Flow_Priority'].clip(lower=controls['fresh_food_flow_priority'])

    freq_map = {
        'Daily': controls['freq_daily'],
        '4x/Week': controls['freq_4x_week'],
        'Weekly': controls['freq_weekly'],
        'Biweekly': controls['freq_biweekly'],
        'Monthly': controls['freq_monthly'],
        'Ad Hoc': controls['freq_adhoc'],
    }
    type_map = {
        'Bulk': controls['type_bulk'],
        'Food': controls['type_food'],
        'Meat': controls['type_meat'],
        'Fragile': controls['type_fragile'],
    }

    df['Inbound_Score'] = df['Inbound_Freq'].apply(lambda x: compute_flow_score(x, freq_map))
    df['Outbound_Score'] = df['Outbound_Freq'].apply(lambda x: compute_flow_score(x, freq_map))
    df['Qty_Received_Score'] = df['Qty_Received'].apply(qty_score)
    df['Qty_Held_Score'] = df['Qty_Held'].apply(qty_score)
    df['ItemType_Score'] = df['Item_Type'].apply(lambda x: compute_item_type_score(x, type_map))

    cube_height_ft = controls['cube_height_ft']
    cube_width_ft = controls['cube_width_in'] / 12
    cube_depth_ft = controls['cube_depth_in'] / 12
    cube_vol = cube_height_ft * cube_width_ft * cube_depth_ft
    df['Cube_Volume_ft3'] = cube_vol

    type_volume = {
        'Bulk': controls['vol_bulk'],
        'Food': controls['vol_food'],
        'Meat': controls['vol_meat'],
        'Fragile': controls['vol_fragile'],
    }
    df['Item_Required_Volume_ft3'] = df['Item_Type'].map(type_volume).fillna(controls['vol_food'])
    df['Fit_Check'] = (df['Item_Required_Volume_ft3'] <= df['Cube_Volume_ft3']).map({True: 'Fits', False: 'Review'})
    df['Cube_Utilization_%'] = ((df['Item_Required_Volume_ft3'] / df['Cube_Volume_ft3']) * 100).round(1)

    df['Storage_Priority_Score'] = (
        df['Usage_Rate'] * controls['w_usage']
        + df['Flow_Priority'] * controls['w_flow']
        + df['Qty_Received_Score'] * controls['w_qty_received']
        + df['Qty_Held_Score'] * controls['w_qty_held']
        + df['ItemType_Score'] * controls['w_item_type']
        + df[['Inbound_Score', 'Outbound_Score']].max(axis=1) * controls['w_freq']
    ).round(2)

    df['Preferred_Area'] = [infer_preferred_area(i, c, t) for i, c, t in zip(df['Item'], df['Category'], df['Item_Type'])]
    df = df.sort_values('Storage_Priority_Score', ascending=False).reset_index(drop=True)
    df['Priority_Rank'] = range(1, len(df) + 1)

    def bucket(rank):
        if rank <= controls['premium_n']:
            return 'Premium'
        if rank <= controls['premium_n'] + controls['standard_n']:
            return 'Standard'
        return 'Reserve'

    df['Capacity_Bucket'] = df['Priority_Rank'].apply(bucket)

    def base_zone(row):
        if row['Capacity_Bucket'] == 'Premium':
            return 'Row 1 (Front)'
        if row['Capacity_Bucket'] == 'Standard':
            return 'Row 2 (Middle)'
        return 'Row 3 (Back)'

    df['Base_Grid_Zone'] = df.apply(base_zone, axis=1)

    def access_band(row):
        flow = safe_num(row['Flow_Priority'])
        usage = safe_num(row['Usage_Rate'])
        held = safe_num(row['Qty_Held'])
        item_type = str(row['Item_Type']).strip()
        bucket_name = row['Capacity_Bucket']
        bulky = item_type == 'Bulk' or held >= controls['bulk_cutoff']

        if bucket_name == 'Premium':
            if flow >= 4 or usage >= 4:
                return 'Front'
            return 'Middle'
        if bucket_name == 'Standard':
            if flow >= 4:
                return 'Front'
            if bulky:
                return 'Middle'
            if flow >= 2.5 or usage >= 2.5:
                return 'Middle'
            return 'Back'
        if bulky:
            return 'Middle'
        return 'Back'

    df['Smart_Access_Band'] = df.apply(access_band, axis=1)

    def smart_adjustment(row):
        if row['Base_Grid_Zone'] == 'Row 1 (Front)' and row['Smart_Access_Band'] == 'Front':
            return 'No override'
        if row['Base_Grid_Zone'] == 'Row 2 (Middle)' and row['Smart_Access_Band'] == 'Middle':
            return 'No override'
        if row['Base_Grid_Zone'] == 'Row 3 (Back)' and row['Smart_Access_Band'] == 'Back':
            return 'No override'
        if row['Smart_Access_Band'] == 'Front':
            return 'Move forward'
        if row['Smart_Access_Band'] == 'Middle':
            return 'Keep accessible'
        return 'Move deeper'

    df['Smart_Adjustment'] = df.apply(smart_adjustment, axis=1)
    df['Band_Rank'] = df.groupby('Smart_Access_Band').cumcount() + 1

    band_positions = {
        'Front': ['A1-Front', 'B1-Front', 'C1-Front'],
        'Middle': ['A2-Middle', 'B2-Middle', 'C2-Middle'],
        'Back': ['A3-Back', 'B3-Back', 'C3-Back'],
    }

    def smart_location(row):
        positions = band_positions[row['Smart_Access_Band']]
        pos = positions[(int(row['Band_Rank']) - 1) % len(positions)]
        return f"{row['Preferred_Area']}-{pos}"

    df['Hybrid_Location'] = df.apply(smart_location, axis=1)

    def reason(row):
        return f"{row['Capacity_Bucket']} item assigned to {row['Smart_Access_Band']} access band based on score, flow, usage, and handling characteristics."

    def ai_rec(row):
        notes = []
        if row['Item_Type'] == 'Bulk':
            notes.append('Bulky item kept away from deep-back handling when possible.')
        if row['Flow_Priority'] >= 4:
            notes.append('High-flow item prioritized for faster access.')
        if row['Fit_Check'] != 'Fits':
            notes.append('Review physical fit before final slotting.')
        if row['Qty_Held'] >= controls['bulk_cutoff']:
            notes.append('High on-hand quantity needs easier replenishment access.')
        if row['Preferred_Area'] == 'ADDITIONS':
            notes.append('Use as overflow or support zone rather than quantified pallet area.')
        if controls['fresh_food_max_days'] <= 3 and str(row['Item']).strip().lower() in {'chicken leg quarters', 'ground beef'}:
            notes.append('Fresh item should move quickly under the 3-day stay rule.')
        if row['Smart_Access_Band'] == 'Middle':
            notes.append('Assigned to middle access band to balance accessibility and space discipline.')
        return ' '.join(notes) if notes else 'Maintain balanced access with standard replenishment rules.'

    df['Hybrid_Reason'] = df.apply(reason, axis=1)
    df['AI_Recommendation'] = df.apply(ai_rec, axis=1)
    return df

st.sidebar.header('Workbook')
uploaded = st.sidebar.file_uploader('Upload workbook', type=['xlsx'])
local_files = sorted([p.name for p in Path('.').glob('*.xlsx') if not p.name.startswith('~$')])
selected_local = None
if local_files:
    selected_local = st.sidebar.selectbox('Or choose workbook from folder', ['-- none --'] + local_files)

file_bytes = None
workbook_label = None
if uploaded is not None:
    file_bytes = uploaded.getvalue()
    workbook_label = uploaded.name
elif selected_local and selected_local != '-- none --':
    file_bytes = Path(selected_local).read_bytes()
    workbook_label = selected_local

st.sidebar.divider()
st.sidebar.header('Dynamic Controls')
w_usage = st.sidebar.slider('Usage Weight', 0.5, 4.0, 2.0, 0.1)
w_flow = st.sidebar.slider('Flow Weight', 0.5, 4.0, 2.0, 0.1)
w_qty_received = st.sidebar.slider('Qty Received Weight', 0.0, 3.0, 1.2, 0.1)
w_qty_held = st.sidebar.slider('Qty Held Weight', 0.0, 3.0, 0.8, 0.1)
w_item_type = st.sidebar.slider('Item Type Weight', 0.0, 3.0, 1.0, 0.1)
w_freq = st.sidebar.slider('Frequency Weight', 0.0, 3.0, 1.0, 0.1)

st.sidebar.subheader('Item Type Scoring')
type_bulk = st.sidebar.slider('Bulk Type Score', 1.0, 6.0, 5.0, 0.5)
type_food = st.sidebar.slider('Food Type Score', 1.0, 6.0, 3.0, 0.5)
type_meat = st.sidebar.slider('Meat Type Score', 1.0, 6.0, 4.0, 0.5)
type_fragile = st.sidebar.slider('Fragile Type Score', 1.0, 6.0, 3.0, 0.5)

st.sidebar.subheader('Bucket Settings')
premium_n = st.sidebar.slider('Premium items', 1, 10, 5, 1)
standard_n = st.sidebar.slider('Standard items', 1, 12, 6, 1)

st.sidebar.subheader('Business Rules')
peanut_flow_priority = st.sidebar.slider('Peanut Butter Flow Priority', 1.0, 5.0, 5.0, 0.5)
fresh_food_max_days = st.sidebar.slider('Fresh Food Max Days', 1, 7, 3, 1)
fresh_food_flow_priority = st.sidebar.slider('Fresh Food Flow Priority', 1.0, 5.0, 5.0, 0.5)
bulk_cutoff = st.sidebar.number_input('Bulk Cutoff (Qty Held)', min_value=100, max_value=5000, value=1000, step=50)

st.sidebar.subheader('Cube Measurements')
cube_height_ft = st.sidebar.slider('Cube Height (ft)', 5.0, 6.5, 5.67, 0.01)
cube_width_in = st.sidebar.number_input('Cube Width (in)', min_value=20, max_value=80, value=40, step=1)
cube_depth_in = st.sidebar.number_input('Cube Depth (in)', min_value=20, max_value=80, value=48, step=1)

st.sidebar.subheader('Volume Assumptions')
vol_bulk = st.sidebar.slider('Bulk Volume (ft³)', 20.0, 100.0, 72.0, 1.0)
vol_food = st.sidebar.slider('Food Volume (ft³)', 10.0, 60.0, 35.0, 1.0)
vol_meat = st.sidebar.slider('Meat Volume (ft³)', 10.0, 80.0, 48.0, 1.0)
vol_fragile = st.sidebar.slider('Fragile Volume (ft³)', 10.0, 60.0, 28.0, 1.0)

st.sidebar.subheader('Frequency Mapping')
freq_daily = st.sidebar.slider('Daily', 1.0, 6.0, 5.0, 0.5)
freq_4x_week = st.sidebar.slider('4x/Week', 1.0, 6.0, 4.5, 0.5)
freq_weekly = st.sidebar.slider('Weekly', 1.0, 6.0, 4.0, 0.5)
freq_biweekly = st.sidebar.slider('Biweekly', 1.0, 6.0, 3.0, 0.5)
freq_monthly = st.sidebar.slider('Monthly', 1.0, 6.0, 2.0, 0.5)
freq_adhoc = st.sidebar.slider('Ad Hoc', 1.0, 6.0, 1.0, 0.5)

if 'new_items' not in st.session_state:
    st.session_state['new_items'] = []

st.sidebar.divider()
st.sidebar.header('Add New Item')
with st.sidebar.form('new_item_form', clear_on_submit=True):
    ni_item = st.text_input('Item Name')
    ni_category = st.selectbox('Category', ['Food', 'Water', 'Meat', 'Other'])
    ni_type = st.selectbox('Item Type', ['Food', 'Bulk', 'Meat', 'Fragile'])
    ni_usage = st.slider('Usage Rate', 1.0, 5.0, 3.0, 0.5)
    ni_received = st.number_input('Qty Received', min_value=0, value=100, step=10)
    ni_held = st.number_input('Qty Held', min_value=0, value=100, step=10)
    ni_flow = st.slider('Flow Priority', 1.0, 5.0, 3.0, 0.5)
    ni_inbound = st.selectbox('Inbound Frequency', ['Daily', '4x/Week', 'Weekly', 'Biweekly', 'Monthly', 'Ad Hoc'])
    ni_outbound = st.selectbox('Outbound Frequency', ['Daily', '4x/Week', 'Weekly', 'Biweekly', 'Monthly', 'Ad Hoc'])
    ni_current = st.text_input('Current Location', value='B2-Middle')
    submitted = st.form_submit_button('Add Item')
    if submitted and ni_item.strip():
        st.session_state['new_items'].append({
            'Item': ni_item,
            'Category': ni_category,
            'Item_Type': ni_type,
            'Usage_Rate': ni_usage,
            'Qty_Received': ni_received,
            'Qty_Held': ni_held,
            'Flow_Priority': ni_flow,
            'Inbound_Freq': ni_inbound,
            'Outbound_Freq': ni_outbound,
            'Current_Location': ni_current,
        })
        st.success(f'{ni_item} added.')

if st.sidebar.button('Clear Added Items'):
    st.session_state['new_items'] = []

st.title('📦 SVDP Control Tower')
st.caption('Hybrid dashboard: Grid + Intelligence')

if file_bytes is None:
    st.info('Upload the Excel workbook or choose one from the folder on the left.')
    st.stop()

st.success(f'Workbook loaded: {workbook_label}')

raw_sheets = load_workbook_tables(file_bytes)
slotting_raw, monthly_flow, clean_tx, warehouse_capacity_raw = read_known_tables(raw_sheets)
capacity_df = clean_capacity(warehouse_capacity_raw)
additions_present = extract_additions_note(warehouse_capacity_raw)

new_items_df = pd.DataFrame(st.session_state['new_items'])

controls = {
    'w_usage': w_usage,
    'w_flow': w_flow,
    'w_qty_received': w_qty_received,
    'w_qty_held': w_qty_held,
    'w_item_type': w_item_type,
    'w_freq': w_freq,
    'type_bulk': type_bulk,
    'type_food': type_food,
    'type_meat': type_meat,
    'type_fragile': type_fragile,
    'premium_n': premium_n,
    'standard_n': standard_n,
    'peanut_flow_priority': peanut_flow_priority,
    'fresh_food_max_days': fresh_food_max_days,
    'fresh_food_flow_priority': fresh_food_flow_priority,
    'bulk_cutoff': bulk_cutoff,
    'cube_height_ft': cube_height_ft,
    'cube_width_in': cube_width_in,
    'cube_depth_in': cube_depth_in,
    'vol_bulk': vol_bulk,
    'vol_food': vol_food,
    'vol_meat': vol_meat,
    'vol_fragile': vol_fragile,
    'freq_daily': freq_daily,
    'freq_4x_week': freq_4x_week,
    'freq_weekly': freq_weekly,
    'freq_biweekly': freq_biweekly,
    'freq_monthly': freq_monthly,
    'freq_adhoc': freq_adhoc,
}

model_df = build_model(slotting_raw, controls, new_items_df)

total_spend = clean_tx['Net ($)'].apply(safe_num).sum() if 'Net ($)' in clean_tx.columns else 0
unique_suppliers = clean_tx['Supplier'].nunique() if 'Supplier' in clean_tx.columns else 0
cap_increase = capacity_df['Increase'].sum()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'Executive', 'Goods Flow', 'Dynamic Slotting', 'Reallocation Plan', 'AI Recommendations', 'Scenario Lab'
])

with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Total Spend', f'${total_spend:,.0f}')
    c2.metric('Capacity Increase', f'{cap_increase:,.0f} pallets')
    c3.metric('Items Evaluated', len(model_df))
    c4.metric('Premium Items', int((model_df['Capacity_Bucket'] == 'Premium').sum()))
    c5.metric('Suppliers', unique_suppliers)

    st.subheader('Capacity Increase Breakdown')
    st.dataframe(capacity_df, hide_index=True, use_container_width=True)
    if additions_present:
        st.info('ADDITIONS is included as a qualitative support zone. It is not counted in the quantified pallet increase because no pallet counts were defined for it.')

    fig_cap = px.bar(capacity_df, x='Layout Area', y='Increase', text='Increase', title='Capacity Increase by Warehouse Area')
    st.plotly_chart(fig_cap, use_container_width=True)

    left, right = st.columns(2)
    with left:
        bucket_counts = model_df['Capacity_Bucket'].value_counts().reset_index()
        bucket_counts.columns = ['Bucket', 'Count']
        fig = px.pie(bucket_counts, names='Bucket', values='Count', title='Capacity Bucket Distribution')
        st.plotly_chart(fig, use_container_width=True)
    with right:
        area_counts = model_df['Preferred_Area'].value_counts().reset_index()
        area_counts.columns = ['Area', 'Count']
        fig = px.bar(area_counts, x='Area', y='Count', text='Count', title='Recommended Distribution by Warehouse Area')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader('Goods Flow')
    mf = monthly_flow.copy()
    mf = mf[mf['Month'].notna()].copy()
    for col in ['Total Inbound Lbs', 'Net Usable Lbs', 'RTV Lbs', 'Garbage Out Lbs']:
        if col in mf.columns:
            mf[col] = mf[col].apply(safe_num)
    if {'Total Inbound Lbs', 'Net Usable Lbs'}.issubset(mf.columns):
        mf['Yield %'] = (mf['Net Usable Lbs'] / mf['Total Inbound Lbs']).replace([math.inf, -math.inf], 0).fillna(0) * 100

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Inbound', x=mf['Month'], y=mf['Total Inbound Lbs']))
        fig.add_trace(go.Bar(name='Net Usable', x=mf['Month'], y=mf['Net Usable Lbs']))
        fig.update_layout(barmode='group', title='Inbound vs Net Usable')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(mf, x='Month', y='Yield %', markers=True, title='Net Usable Yield %')
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(mf, hide_index=True, use_container_width=True)

with tab3:
    st.subheader('Dynamic Slotting (Grid + Intelligence)')
    a1, a2, a3 = st.columns(3)

    area_options = sorted([str(x) for x in model_df['Preferred_Area'].dropna().astype(str).unique().tolist() if str(x).strip()])
    item_type_options = sorted([str(x) for x in model_df['Item_Type'].dropna().astype(str).unique().tolist() if str(x).strip() and str(x).lower() != 'nan'])

    area_filter = a1.multiselect('Preferred Area', area_options, default=area_options)
    bucket_filter = a2.multiselect('Capacity Bucket', ['Premium', 'Standard', 'Reserve'], default=['Premium', 'Standard', 'Reserve'])
    item_filter = a3.multiselect('Item Type', item_type_options, default=item_type_options)

    view = model_df[
        model_df['Preferred_Area'].astype(str).isin(area_filter) &
        model_df['Capacity_Bucket'].astype(str).isin(bucket_filter) &
        model_df['Item_Type'].astype(str).isin(item_filter)
    ].copy()

    left, right = st.columns(2)
    with left:
        fig = px.bar(view, x='Item', y='Storage_Priority_Score', color='Capacity_Bucket', text='Storage_Priority_Score', title='Storage Priority Score by Item')
        st.plotly_chart(fig, use_container_width=True)
    with right:
        patterns = view['Hybrid_Location'].str.extract(r'-(A1|B1|C1|A2|B2|A3|B3|C2|C3)-')
        pc = patterns[0].value_counts().reindex(['A1','B1','C1','A2','B2','A3','B3','C2','C3'], fill_value=0).reset_index()
        pc.columns = ['Pattern','Count']
        fig = px.bar(pc, x='Pattern', y='Count', title='Hybrid Location Mix')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Warehouse Grid Utilization')
    grid_map = view.copy()
    grid_map['Grid_Row'] = grid_map['Hybrid_Location'].str.extract(r'-(A|B|C)\d-')
    grid_map['Grid_Col'] = grid_map['Hybrid_Location'].str.extract(r'-[ABC]([123])-')
    pivot = grid_map.groupby(['Grid_Row', 'Grid_Col']).size().reset_index(name='Count')
    if not pivot.empty:
        fig = px.density_heatmap(pivot, x='Grid_Col', y='Grid_Row', z='Count', text_auto=True, title='Grid Utilization Heatmap')
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(view[[
        'Item','Category','Item_Type','Usage_Rate','Flow_Priority','Qty_Received','Qty_Held',
        'Storage_Priority_Score','Priority_Rank','Capacity_Bucket','Preferred_Area',
        'Base_Grid_Zone','Smart_Access_Band','Smart_Adjustment',
        'Current_Location','Hybrid_Location','Fit_Check','Cube_Utilization_%'
    ]], hide_index=True, use_container_width=True)

with tab4:
    st.subheader('Reallocation Plan')
    plan = model_df[[
        'Item','Current_Location','Capacity_Bucket','Base_Grid_Zone','Smart_Adjustment',
        'Hybrid_Location','Hybrid_Reason','AI_Recommendation'
    ]].copy()
    plan.columns = [
        'Item','Current Location','Bucket','Base Grid Zone','Smart Adjustment',
        'Final Location','Model Reason','AI Recommendation'
    ]
    st.dataframe(plan, hide_index=True, use_container_width=True)
    csv = plan.to_csv(index=False).encode('utf-8')
    st.download_button('Download Reallocation Plan CSV', data=csv, file_name='svdp_reallocation_plan.csv', mime='text/csv')

with tab5:
    st.subheader('AI Recommendations')
    item = st.selectbox('Select item', model_df['Item'].tolist())
    row = model_df[model_df['Item'] == item].iloc[0]
    a, b = st.columns(2)
    with a:
        st.write(f"**Item:** {row['Item']}")
        st.write(f"**Current Location:** {row['Current_Location']}")
        st.write(f"**Base Grid Zone:** {row['Base_Grid_Zone']}")
        st.write(f"**Smart Access Band:** {row['Smart_Access_Band']}")
        st.write(f"**Smart Adjustment:** {row['Smart_Adjustment']}")
        st.write(f"**Final Location:** {row['Hybrid_Location']}")
        st.write(f"**Preferred Area:** {row['Preferred_Area']}")
        st.write(f"**Bucket:** {row['Capacity_Bucket']}")
        st.write(f"**Priority Score:** {row['Storage_Priority_Score']}")
    with b:
        st.success(row['AI_Recommendation'])
        st.info(row['Hybrid_Reason'])

    st.subheader('Project Coverage')
    coverage = pd.DataFrame([
        {'Requirement': 'Cube utilization analysis', 'Covered by': 'Cube volume, fit check, utilization %'},
        {'Requirement': 'Bin classification', 'Covered by': 'Premium / Standard / Reserve bucket logic'},
        {'Requirement': 'Slotting strategy', 'Covered by': 'Grid + Intelligence location assignment'},
        {'Requirement': 'Goods flow', 'Covered by': 'Goods Flow tab'},
        {'Requirement': 'Warehouse zoning', 'Covered by': 'Preferred area allocation'},
        {'Requirement': 'JIT-inspired flow logic', 'Covered by': 'AI recommendations + frequency / freshness rules'},
    ])
    st.dataframe(coverage, hide_index=True, use_container_width=True)

with tab6:
    st.subheader('Scenario Lab')
    st.write('Change controls in the sidebar and watch the model recalculate.')
    st.dataframe(model_df[['Item','Storage_Priority_Score','Priority_Rank','Capacity_Bucket','Hybrid_Location']], hide_index=True, use_container_width=True)
    scenario_df = model_df.copy()
    scenario_df['Qty_Held'] = pd.to_numeric(scenario_df['Qty_Held'], errors='coerce').fillna(0).clip(lower=0)
    scenario_df['Qty_Received'] = pd.to_numeric(scenario_df['Qty_Received'], errors='coerce').fillna(0).clip(lower=0)
    scenario_df['Storage_Priority_Score'] = pd.to_numeric(scenario_df['Storage_Priority_Score'], errors='coerce').fillna(0)
    fig = px.scatter(scenario_df, x='Qty_Held', y='Storage_Priority_Score', size='Qty_Received', color='Capacity_Bucket', hover_name='Item', title='Scenario View: Score vs Qty Held')
    st.plotly_chart(fig, use_container_width=True)
