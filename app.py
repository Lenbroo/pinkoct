import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np

# --- 1. Page Setup and Design ---

# Set the page to "wide" layout and use a dark theme for plots
st.set_page_config(page_title="Sales Dashboard", layout="wide")
pio.templates.default = "plotly_dark"

# Define the target packages
TARGET_PACKAGES = [
    'PREMIUM PLUS WOMENS HEALTH SCREENING',
    'WOMENS COMPREHENSIVE HEALTH PACKAGE'
]

# --- 2. Data Processing Functions (with Caching) ---

@st.cache_data
def load_and_process_data(uploaded_file):
    """
    This function loads and processes the uploaded Excel file.
    It returns 6 DataFrames ready for plotting.
    """
    try:
        df = pd.read_excel(uploaded_file, header=0)
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
        return (pd.DataFrame(),) * 6 # Return 6 empty DataFrames on error

    if len(df.columns) < 6:
        st.error(f"Error: File must contain at least 6 columns. Found {len(df.columns)}.")
        return (pd.DataFrame(),) * 6

    # Split the data
    df_oct = df.iloc[:, 0:3].copy()
    df_oct.columns = ['Date', 'Location', 'Package']
    df_oct['Month'] = 'October'

    df_sep = df.iloc[:, 3:6].copy()
    df_sep.columns = ['Date', 'Location', 'Package']
    df_sep['Month'] = 'September'

    df_all = pd.concat([df_oct, df_sep], ignore_index=True)

    # Clean and filter data
    df_all = df_all.dropna(subset=['Date', 'Package'])

    df_all['Package'] = df_all['Package'].replace(
        'WOMENS COMPREHENSIVE HEALTH SCREENING', 
        'WOMENS COMPREHENSIVE HEALTH PACKAGE'
    )
    df_all = df_all[df_all['Package'].isin(TARGET_PACKAGES)].copy()

    df_all['Date'] = pd.to_datetime(df_all['Date'], format='%d/%m/%Y', errors='coerce')
    df_all = df_all.dropna(subset=['Date'])

    if df_all.empty:
        return (pd.DataFrame(),) * 6 # Return empty data if nothing found

    df_all['DayOfWeekName'] = df_all['Date'].dt.day_name()
    df_all['DayOfWeekNum'] = df_all['Date'].dt.weekday
    df_all = df_all.sort_values(by='Date')

    # --- Prepare DataFrames for plotting ---

    # 1. For KPI Table
    sales_summary = df_all.groupby(['Month', 'Package']).size().reset_index(name='TotalSales')
    summary_pivot = sales_summary.pivot(
        index='Package', columns='Month', values='TotalSales'
    ).reindex(columns=['September', 'October']).fillna(0).astype(int)
    summary_pivot['% Change'] = ((summary_pivot['October'] - summary_pivot['September']) / summary_pivot['September']) * 100
    summary_pivot['% Change'] = summary_pivot['% Change'].replace([np.inf, -np.inf], np.nan)

    # 2. For DOW Heatmap
    dow_sales = df_all.groupby(['Month', 'DayOfWeekName', 'DayOfWeekNum']).size().reset_index(name='TotalSales')
    dow_sales = dow_sales.sort_values(by='DayOfWeekNum')
    day_order = dow_sales['DayOfWeekName'].unique()
    dow_pivot = dow_sales.pivot(
        index='Month', columns='DayOfWeekName', values='TotalSales'
    ).fillna(0).astype(int)
    dow_pivot = dow_pivot.reindex(columns=day_order)
    dow_pivot = dow_pivot.reindex(index=['September', 'October'])

    # 3. For Daily Trend
    daily_sales = df_all.groupby(['Date', 'Package']).size().reset_index(name='SalesCount')

    # 4. For Location Chart
    location_sales = df_all.groupby(['Month', 'Package', 'Location']).size().reset_index(name='TotalSales')

    return df_all, summary_pivot, sales_summary, dow_pivot, daily_sales, location_sales

# --- 3. Chart Creation Functions ---

def create_kpi_table(summary_pivot):
    summary_display = summary_pivot.reset_index()
    summary_display['% Change'] = summary_display['% Change'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A (New)")
    
    fig_table = go.Figure(data=[go.Table(
        header=dict(values=list(summary_display.columns),
                    fill_color='#2a3f5f', # Darker header
                    align='left',
                    font=dict(color='white', size=14)),
        cells=dict(values=[summary_display[col] for col in summary_display.columns],
                   fill_color='#1e2a40', # Lighter cells
                   align='left',
                   font=dict(color='white', size=12),
                   height=30)
    )])
    fig_table.update_layout(title_text="Key Performance Indicators (KPI) Summary Table")
    return fig_table

def create_total_sales_chart(sales_summary):
    fig = px.bar(
        sales_summary,
        x='Package',
        y='TotalSales',
        color='Month',
        barmode='group',
        title='Campaign Impact: Total Sales (Sep vs. Oct)',
        labels={'TotalSales': 'Total Number of Packages Sold'},
        text_auto=True
    )
    fig.update_layout(xaxis_title=None)
    return fig

def create_dow_heatmap(dow_pivot):
    # Use the custom heatmap code as requested
    fig = go.Figure(data=go.Heatmap(
        z=dow_pivot.values,
        x=dow_pivot.columns,
        y=dow_pivot.index,
        colorscale='Reds', # Red color scale
        colorbar=dict(title='Total Sales'),
        text=dow_pivot.values, # Numbers inside
        texttemplate="%{text}", # Number format
        textfont={"size":12, "color":"black"} # Text color (black for light red)
    ))
    fig.update_layout(
        title_text='Sales Volume by Day of Week (Sep vs. Oct)',
        xaxis_title='Day of Week',
        yaxis_title='Month',
        xaxis=dict(side="top"),
    )
    return fig

def create_daily_trend_chart(daily_sales, df_all):
    fig = px.line(
        daily_sales,
        x='Date',
        y='SalesCount',
        color='Package',
        title='Daily Sales Trend (Sep & Oct)',
        labels={'SalesCount': 'Number of Sales'},
        markers=True
    )
    first_oct_date = df_all[df_all['Month'] == 'October']['Date'].min()
    if pd.notna(first_oct_date):
        fig.add_vline(
            x=first_oct_date, 
            line_width=2, 
            line_dash="dash", 
            line_color="red"
        )
    return fig

def create_location_chart(location_sales):
    fig = px.bar(
        location_sales,
        x='Location',
        y='TotalSales',
        color='Month',
        barmode='group',
        facet_row='Package',
        title='Sales Performance by Location',
        labels={'TotalSales': 'Total Sales'},
        height=600
    )
    fig.update_xaxes(tickangle=45)
    return fig

# --- 4. Build the App Interface ---

st.title("📈 Women's Health Package Sales Dashboard")
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your Excel file (e.g., Book11.xlsx) to start analysis", type=["xlsx"])

if uploaded_file is not None:
    # Load and process the data
    df_all, summary_pivot, sales_summary, dow_pivot, daily_sales, location_sales = load_and_process_data(uploaded_file)
    
    if df_all.empty:
        st.warning("No data found for the specified packages in the uploaded file. Please check the file.")
    else:
        # Display the KPI table first
        st.header("📊 Key Performance Indicators (KPIs) Summary")
        fig_table = create_kpi_table(summary_pivot)
        st.plotly_chart(fig_table, use_container_width=True)
        
        st.markdown("---")
        
        # Split screen for main charts
        st.header("🔍 Campaign Comparative Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Total Sales (September vs. October)")
            fig_total_sales = create_total_sales_chart(sales_summary)
            st.plotly_chart(fig_total_sales, use_container_width=True)
            
        with col2:
            st.subheader("Sales by Day of Week (DOW) Analysis")
            fig_dow_heatmap = create_dow_heatmap(dow_pivot)
            st.plotly_chart(fig_dow_heatmap, use_container_width=True)

        st.markdown("---")
        st.header("📈 Sales Trends and Location Analysis")
        
        # Display longer charts
        st.subheader("Daily Sales Trend (September & October)")
        fig_daily_trend = create_daily_trend_chart(daily_sales, df_all)
        st.plotly_chart(fig_daily_trend, use_container_width=True)
        
        st.subheader("Sales Performance by Location")
        fig_location = create_location_chart(location_sales)
        st.plotly_chart(fig_location, use_container_width=True)

else:
    st.info("Please upload an Excel file to view the dashboard.")
    st.image("https://placehold.co/1200x400/1e2a40/ffffff?text=Your+Dashboard+is+Waiting...", use_column_width=True)

