
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import seaborn as sns
import pandas_datareader
import pydst 
import seaborn as sns 

class dataprojectclass:

    #Defining a function to extract data from DST
    def get_data(self, table, variables_list):
        Dst = pydst.Dst(lang='en') # setup data loader with the langauge 'english'
        
        plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
        plt.rcParams.update({'font.size': 14})
        tables = Dst.get_tables(subjects=['2'])
        # Retrieving subjets, tables and variables
        Dst.get_subjects()
        tables = Dst.get_tables(subjects=['4'])
        tables[tables.id == table]
        indk_vars = Dst.get_variables(table_id=table)
        # Retriving data from specific tables
        variables = variables_list
        df = Dst.get_data(table_id = table, variables=variables)
        return df

    #Defining a function to convert a string to float by first removing part of string
    def string_to_float(self, variabel, strip_string):
        string = variabel.map(lambda x: x.rstrip(strip_string))
        float = string.astype('float64')
        return float
        
    #Defining a function to drop certain observations
    def drop_obs(self, df, column, value, equality = "not_equal"):
        if equality == "not_equal":
            not_dropped = df.drop(df[df[column] != value].index, inplace = True)
        return not_dropped

    #Defining a function to drop variables/columns
    def drop_variables(self, df, columns):
        not_dropped = df.drop(columns=columns, inplace = True)
        return not_dropped


    def figure_region(self, education, df):

        fig = df[(df['EDUCATION'] == education) & (df['Date'].isin([2010, 2022]))]
            
        # Create bar plot
        ax = sns.barplot(data=fig, x='REGION', y='ratio', hue='Date', ci = None)
            
        # Rotate x-axis labels
        #plt.xticks(rotation=90)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # Set axis labels and title
        ax.set_xlabel('')
        ax.set_ylabel('Ratio (%)')

        long_title = f'Ratio of population with {education} as their highest completed education level by region and year'
        wrapped_title = textwrap.fill(long_title, width=60) 
        plt.title(wrapped_title)
        plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
        #plt.text(0, -40, "Note: Population aged 25 - 39 years", ha='center', va='center', fontsize=10, fontweight='bold')

        # Return the plot object
        return ax



    def subplot(self, map_dict, df2, key, education, max_value = None):
        # Create a new figure with a grid layout (1 row, 2 columns)
        fig, axs = plt.subplots(1, 2, figsize=(15, 8))

        # Plot 'a' on the first subplot
        map_dict[key].plot(ax=axs[0], column='ratio', cmap='coolwarm', legend=True,legend_kwds={'label': "Ratio", 'orientation': "horizontal"}, vmin=0, vmax = max_value)
        axs[0].grid(False)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title('Across municipalities in 2022')

        # Plot 'b' on the second subplot
        self.figure_region(education, df2).plot(ax=axs[1], column='ratio', cmap='coolwarm', legend=True)
        axs[1].set_title('Across regions and time')

        # Adjust spacing between subplots
        plt.tight_layout()

        # Add collective title
        plt.suptitle(f"Ratio of Population with {education} as the Highest Completed Level of Education", fontsize=16, x=0.45, y=1.1)

        # Show the plot
        plt.show()


    def ratio(self, map_dict, municipality):
        selected_municipality = map_dict[map_dict['Municipalities'] == municipality]
        ratio = selected_municipality['ratio'].values[0]
        print = str(f'The ratio for {municipality} was {ratio:.1f} pct. in 2022.')
        return print