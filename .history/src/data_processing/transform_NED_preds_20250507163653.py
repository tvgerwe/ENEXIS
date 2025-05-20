df_NED_preds_CSV['validfrom'] = pd.to_datetime(df['validfrom'])
df['validto'] = pd.to_datetime(df['validto'])
df['lastupdate'] = pd.to_datetime(df['lastupdate'])