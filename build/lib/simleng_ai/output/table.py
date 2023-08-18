class Table_results:
    """ Show table results."""
    """
      0:params
      1:residuals
      2:fitted_values
      3:y_calibrated
      4:y_estimated
      5:z_score
      6:confusion_matrix
      7:iv_l
      8:iv_u
      9:vif
     10:acc
     11:ppv
     12:acc_mean
     13:ppv_mean
    """

    def __init__(self,var,table,floatfmt,style,title,width):

            self.var=var
            self.table=table
            self.floatfmt=floatfmt
            self.tablefmt=style
            self.title=title
            self.width=width


    def print_table(self):
        from resources.output import table 
        print_table=["params_table","residuals_table","fitted_values_table","y_calibrated_table", \
                "y_estimated_table","z_score_table","confusion_matrix","iv_l_table","iv_u_table",\
                "vif_table","acc_table","ppv_table","acc_mean_table","ppv_mean_table"]
        self.table_name=print_table[self.var]
        if self.table_name in print_table:
            return table(self.table,self.floatfmt,self.tablefmt,self.title,self.width)
        else:
            return print("Error in print_table name")
        outer_name ='print_' + str(self.table_name)
        # Get the method from 'self'. Default to a lambda.
        outer= getattr(self,outer_name,lambda:"Invalid Method selected")
        # call the strategie as we return it
        return outer()
