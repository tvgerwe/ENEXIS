CSV files contains fetched price predictions, obtained up till the date mentioned.
x y and subarray are values directly obtained from the API
the code for that is in workspaces - twan, and is named get_prices_ipynb
other collumns are added, step by step, as coded in proces_prices.ipynb
which is located in the same folder.
there additional collumns are:
timestamp = UTC timestamp of moment of fetch from API
date_timestamp = date deducted from 'timestamp'
hour_timestamp = hour deducted from 'timestamp'. round down to full hour
date_time = UTC datetime conversion from 'x', showing the hour for which price point is predicted (or published in case of subarray == 0)
Price = electricity price including VAT and energy tax. obtained through conversion from 'y' value. 