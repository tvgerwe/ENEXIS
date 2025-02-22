"""
Purpose:    Init file - Utilities Pieter
Author:     Pieter Overdevest
Date:       2024-02-09
"""

# Set version number
__version__ = "1.0.0"

# Definition of number to demo.
# n_demo = 10

# Import modules - By using this approach we can import the functions
# in to the Jupyter Notebook and there is no need to refer to the module
# name. In the Jupyter Notebook we can write:
# "from utils_pieter import f_info"
# and use the f_info function in the Jupyter Notebook.

# In case we exclude the lines below, we have to state:
# "from utils_pieter.f_info import f_info"
# in order to use said function in the Jupyter Notebook.

# There is no need use the same name for the function and the module
# (Python file). Here, I have use one module per function.

# If you have multiple functions that are related you can put them
# in the same module. Below you could write, in 
# from .module_name import f_one
# from .module_name import f_two
# Or in your Jupyter Notebook, you could write:
# from utils_pieter.module_name import f_one
# from utils_pieter.module_name import f_two

from .m_check_na_in_df               import f_check_na_in_df
from .m_check_nonnumeric_in_df       import f_check_nonnumeric_in_df
from .m_clean_up_header_names        import f_clean_up_header_names
from .m_describe                     import f_describe
from .m_get_account_name             import f_get_account_name
from .m_evaluate_results             import f_evaluate_results
from .m_get_filenames_in_folder      import f_get_filenames_in_folder
from .m_get_latest_file              import f_get_latest_file
from .m_heatmap                      import f_heatmap
from .m_info                         import f_info
from .m_join                         import f_join
from .m_now                          import f_now
from .m_plot_scatter_with_trend_grid import f_plot_scatter_with_trend_grid
from .m_plot_scatter_with_trend      import f_plot_scatter_with_trend
from .m_train_test_split             import f_train_test_split
from .m_var_name                     import f_var_name

# Print message - Just for demo purposes.
print("Done!")





