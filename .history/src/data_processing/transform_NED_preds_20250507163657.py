df_NED_preds_CSV['validfrom'] = pd.to_datetime(df['validfrom'])
df_NED_preds_CSV['validto'] = pd.to_datetime(df['validto'])
df_NED_preds_CSV['lastupdate'] = pd.to_datetime(df['lastupdate'])