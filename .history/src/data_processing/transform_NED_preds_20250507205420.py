import pandas as pd
df_NED_preds_CSV['validfrom'] = pd.to_datetime(df['validfrom'])
df_NED_preds_CSV['validto'] = pd.to_datetime(df['validto'])
df_NED_preds_CSV['lastupdate'] = pd.to_datetime(df['lastupdate'])
df_NED_preds_CSV = df_NED_preds_CSV.drop(columns=['@id', 'id', 'point', 'granularity', 'granularitytimezone', 'activity', 'volume'])
import pandas as pd

# Zorg dat je dataframe ingeladen is als df
# Stap 1: Kies de relevante kolommen
df_subset = df_NED_preds_CSV[['validto', 'lastupdate', 'type', 'volume']]

# Stap 2: Pivot de data zodat 'type' kolommen worden en 'volume' de waarden
df_NED_preds_processed = df_subset.pivot_table(
    index=['validto', 'lastupdate'],
    columns='type',
    values='volume',
    aggfunc='sum'  # Of 'first' als je zeker weet dat er maar één waarde per combinatie is
)

# Stap 3: Reset de index om weer een normale dataframe structuur te krijgen
df_NED_preds_processed = df_NED_preds_processed.reset_index()

# Stap 4 (optioneel): Kolomnamen opschonen
df_NED_preds_processed.columns.name = None