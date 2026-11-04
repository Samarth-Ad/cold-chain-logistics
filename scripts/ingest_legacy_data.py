from pathlib import Path
import pandas as pd
import urllib
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv


script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[0]

load_dotenv(project_root / ".env")

data_path = project_root / "data" / "raw"/ "dynamic_supply_chain_logistics_dataset.csv"

db_host = os.getenv("SQL_SERVER_HOST", "localhost")
db_port = os.getenv("SQL_SERVER_PORT", "1433")
db_user = os.getenv("SQL_ADMIN_USER")
db_password = os.getenv("SQL_ADMIN_PASSWORD")

df = pd.read_csv(data_path)
legacy_mapping = {
    'timestamp': 'TS_UTC',
    'vehicle_gps_latitude': 'V_LAT',
    'vehicle_gps_longitude': 'V_LON',
    'iot_temperature': 'IOT_TEMP_VAL_C',
    'cargo_condition_status': 'CGO_COND_CD',
    'risk_classification': 'RISK_CLS_TXT',
    'delay_probability': 'DELAY_PROB_DEC',
    'port_congestion_level': 'PRT_CNG_LVL',
    'route_risk_level': 'RT_RSK_IDX'
}

# Create a new DataFrame with only the legacy columns and rename them according to the mapping
df_legacy = df[list(legacy_mapping.keys())].rename(columns=legacy_mapping)

df_legacy['SYS_INGEST_FLAG'] = 'Y'

connection_string = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={db_host},{db_port};"
    f"DATABASE=master;"
    f"UID={db_user};"
    f"PWD={db_password};"
    f"Encrypt=no;"
    f"TrustServerCertificate=yes;"
)

params = urllib.parse.quote_plus(connection_string)



engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
table_name = 'TBL_SC_FLEET_HIST_RAW'
df_legacy.to_sql(table_name, engine, if_exists='replace', index=False, schema='dbo')

print(f"✅Legacy data ingested into table '{table_name}' in the database.")