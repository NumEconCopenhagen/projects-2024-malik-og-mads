import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output


class CrimePlot:
    def __init__(self, data):
        self.inc_api = data
        self.gender_dropdown = widgets.Dropdown(options=['Men', 'Women'], description='Gender:', value='Men')
        self.year_slider = widgets.IntSlider(min=2017, max=2022, step=1, description='Year:', value=2017)
        self.controls = widgets.HBox([self.gender_dropdown, self.year_slider])

        # Observe changes for both widgets using the same handler
        self.gender_dropdown.observe(self.on_controls_change, names='value')
        self.year_slider.observe(self.on_controls_change, names='value')

    def plot_crime_shares(self, gender, year):
        # Filter data based on gender and the selected year
        total_crimes_year = self.inc_api[(self.inc_api['TID'] == year) & (self.inc_api['IELAND'] == 'Total')]['INDHOLD'].sum()
        
        # Exclude specified countries
        exclude_countries = ['Denmark', 'Other countries, non-western', 'Other countries, total', 'Total']
        
        # Find top 5 countries and excluding specified countries
        top_countries_gender = self.inc_api[(self.inc_api['KOEN'] == gender) & (self.inc_api['TID'] == year) & (~self.inc_api['IELAND'].isin(exclude_countries))].nlargest(5, 'INDHOLD')['IELAND'].tolist()
        
        filtered_data = self.inc_api[(self.inc_api['KOEN'] == gender) & (self.inc_api['TID'] == year) & (self.inc_api['IELAND'].isin(top_countries_gender))].copy()
        
        if total_crimes_year > 0:  # We make sure that the data can not be 0
            filtered_data['Share'] = filtered_data['INDHOLD'] / total_crimes_year * 100
            
            # sorting datanames for the graph so it is alphabetic to the left
            filtered_data = filtered_data.sort_values(by='IELAND')
            
            plt.figure(figsize=(10, 6))
            plt.bar(filtered_data['IELAND'], filtered_data['Share'])
            plt.title(f'Crime Shares Among {gender} in {year}')
            plt.xlabel('Country')
            plt.ylabel('Share of Total Crimes (%)')
            plt.xticks(rotation=45)
            plt.show()

    def on_controls_change(self, change):
        clear_output(wait=True)
        # This code is to ensure that the controls display after clearing
        display(self.controls)
        self.plot_crime_shares(self.gender_dropdown.value, self.year_slider.value)

    def display_controls(self):
        display(self.controls)
        self.plot_crime_shares(self.gender_dropdown.value, self.year_slider.value)


class CrimeSharePlot:
    def __init__(self, data):
        self.inc_api = data
        self.gender_dropdown = widgets.Dropdown(options=['Men', 'Women'], description='Gender:', value='Men')
        self.year_slider = widgets.IntSlider(description='Year', min=2017, max=2022, value=2017)
        self.controls = widgets.VBox([self.gender_dropdown, self.year_slider])

        self.gender_dropdown.observe(self.on_gender_year_selected, names='value')
        self.year_slider.observe(self.on_gender_year_selected, names='value')

    def plot_pie_chart(self, gender, selected_year):
        filtered_data = self.inc_api[(self.inc_api['KOEN'] == gender) & 
                                     (self.inc_api['IELAND'] != 'Total') & 
                                     (self.inc_api['IELAND'] != 'Denmark') & 
                                     (self.inc_api['IELAND'] != 'Other countries, non-western') & 
                                     (self.inc_api['IELAND'] != 'Other countries, total') & 
                                     (self.inc_api['TID'].astype(int) == selected_year)]
        top_countries = filtered_data.groupby('IELAND')['INDHOLD'].sum().nlargest(5)

        plt.figure(figsize=(10, 6))
        top_countries.plot(kind='pie', autopct='%1.1f%%', startangle=90, ylabel='', 
                           title=f'Top 5 Countries by Crime Rate Among {gender} ({selected_year})')
        plt.show()

    def on_gender_year_selected(self, change):
        clear_output(wait=True)
        display(self.controls)
        self.plot_pie_chart(self.gender_dropdown.value, self.year_slider.value)

    def display_controls(self):
        display(self.controls)
        self.plot_pie_chart(self.gender_dropdown.value, self.year_slider.value)

class CrimeDevelopmentPlot:
    def __init__(self, data):
        self.inc_api = data
        self.gender_dropdown = widgets.Dropdown(options=['Men', 'Women'], description='Gender:', value='Men')
        self.gender_dropdown.observe(self.on_gender_selected, names='value')

    def plot_crime_development(self, gender):
        # Filter by gender and year
        filtered_data = self.inc_api[(self.inc_api['KOEN'] == gender) & (self.inc_api['TID'].between(2017, 2022))]
        
        # Define countries based on gender
        countries = []
        if gender == 'Men':
            countries = ["Turkey", "Pakistan", "Iraq", "Syria", "Lebanon"]
        elif gender == 'Women':
            countries = ["Turkey", "Pakistan", "Iraq", "Poland", "Lebanon"]
        
        # Include only the specified countries
        filtered_data = filtered_data[filtered_data['IELAND'].isin(countries)]
        
        # Group by country and year and then sum
        grouped_data = filtered_data.groupby(['IELAND', 'TID'])['INDHOLD'].sum().unstack()
        
        # Plot the data
        plt.figure(figsize=(10, 6))
        grouped_data.T.plot(kind='line', marker='o')
        plt.title(f'Crime Development Among {gender} from 2017 to 2022')
        plt.xlabel('Year')
        plt.ylabel('Number of Crimes')
        plt.grid(True)
        plt.legend(title='Country')
        plt.show()

    def on_gender_selected(self, change):
        clear_output(wait=True)
        display(self.gender_dropdown)
        self.plot_crime_development(change['new'])

    def display_controls(self):
        display(self.gender_dropdown)
        self.plot_crime_development('Men')  # Default display

class PeopleSharesPlot:
    def __init__(self, data):
        self.oprindelse_api = data
        self.year_slider = widgets.IntSlider(min=2017, max=2022, step=1, description='Year:', value=2017)
        self.gender_dropdown = widgets.Dropdown(options=['Men', 'Women'], description='Gender:', value='Men')
        self.controls = widgets.HBox([self.year_slider, self.gender_dropdown])

        self.year_slider.observe(self.on_controls_change, names='value')
        self.gender_dropdown.observe(self.on_controls_change, names='value')

    def plot_people_shares(self, year, gender):
        # Filter for time and gender
        filtered_data = self.oprindelse_api[(self.oprindelse_api['TID'] == year) & 
                                            (self.oprindelse_api['KØN'] == gender)]
        
        # Define countries based on gender
        countries_of_interest = []
        if gender == 'Men':
            countries_of_interest = ['Turkey', 'Pakistan', 'Iraq', 'Syria', 'Lebanon']
        elif gender == 'Women':
            countries_of_interest = ['Turkey', 'Pakistan', 'Iraq', 'Poland', 'Lebanon']
        
        # Focus on the specified countries
        filtered_data = filtered_data[filtered_data['IELAND'].isin(countries_of_interest)]
        
        total_people = filtered_data['INDHOLD'].sum()
        filtered_data['Share'] = (filtered_data['INDHOLD'] / total_people) * 100
        
        # Sort the filtered data by country name alphabetically
        filtered_data = filtered_data.sort_values(by='IELAND')
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(filtered_data['IELAND'], filtered_data['Share'])
        plt.title(f'Share of People from Selected Countries Among {gender} in {year}')
        plt.xlabel('Country')
        plt.ylabel('Share of People (%)')
        plt.xticks(rotation=45)
        plt.show()

    def on_controls_change(self, change):
        clear_output(wait=True)
        display(self.controls)
        self.plot_people_shares(self.year_slider.value, self.gender_dropdown.value)

    def display_controls(self):
        display(self.controls)
        self.plot_people_shares(self.year_slider.value, self.gender_dropdown.value)

class ComparisonPlot:
    def __init__(self, inc_data, oprindelse_data):
        self.inc_api = inc_data
        self.oprindelse_api = oprindelse_data
        self.year_slider = widgets.IntSlider(min=2017, max=2022, step=1, description='Year:', value=2017)
        self.gender_dropdown = widgets.Dropdown(options=['Men', 'Women'], description='Gender:', value='Men')
        self.controls = widgets.HBox([self.year_slider, self.gender_dropdown])

        self.year_slider.observe(self.on_controls_change, names='value')
        self.gender_dropdown.observe(self.on_controls_change, names='value')

    def plot_comparison(self, year, gender):
        fig, ax1 = plt.subplots(figsize=(15, 6))
        
        # Define countries based on gender
        if gender == 'Men':
            countries_of_interest = ['Turkey', 'Pakistan', 'Iraq', 'Syria', 'Lebanon']
        elif gender == 'Women':
            countries_of_interest = ['Turkey', 'Pakistan', 'Iraq', 'Poland', 'Lebanon']
        
        # Plotting people shares on primary y-axis
        self.plot_people_shares(year, gender, ax1, countries_of_interest)
        plt.title(f'Crime Shares and Share of People Among {gender} in {year}')
        
        # Secondary y-axis for crime shares
        ax2 = ax1.twinx()
        
        # Plotting crime shares on secondary y-axis
        self.plot_crime_shares(gender, year, ax2, countries_of_interest)
        
        plt.tight_layout()
        plt.show()

    def plot_crime_shares(self, gender, year, ax, countries_of_interest):
        total_crimes_year = self.inc_api[(self.inc_api['TID'] == year) & (self.inc_api['IELAND'] == 'Total')]['INDHOLD'].sum()
        filtered_data = self.inc_api[(self.inc_api['KOEN'] == gender) & (self.inc_api['TID'] == year) & (self.inc_api['IELAND'].isin(countries_of_interest))].copy()
        
        if total_crimes_year > 0:
            filtered_data.loc[:, 'Share'] = filtered_data['INDHOLD'] / total_crimes_year * 100
            filtered_data = filtered_data.sort_values(by='IELAND')
            ax.bar(filtered_data['IELAND'], filtered_data['Share'], label='Crime Shares', alpha=0.5, color='black')
            ax.set_ylabel('Share of Total Crimes (%)')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, 3.5)
            ax.legend(loc='upper right')

    def plot_people_shares(self, year, gender, ax, countries_of_interest):
        filtered_data = self.oprindelse_api[(self.oprindelse_api['TID'] == year) & 
                                            (self.oprindelse_api['KØN'] == gender) & 
                                            (self.oprindelse_api['IELAND'].isin(countries_of_interest))].copy()
        
        total_people = filtered_data['INDHOLD'].sum()
        if total_people > 0:
            filtered_data.loc[:, 'Share'] = filtered_data['INDHOLD'] / total_people * 100
            filtered_data = filtered_data.sort_values(by='IELAND')
            ax.bar(filtered_data['IELAND'], filtered_data['Share'], label='Share of People', alpha=1, color='green')
            ax.set_ylabel('Share of People (%)')
            ax.set_ylim(0, 3.5)
            ax.legend(loc='upper left')

    def on_controls_change(self, change):
        clear_output(wait=True)
        display(self.controls)
        self.plot_comparison(self.year_slider.value, self.gender_dropdown.value)

    def display_controls(self):
        display(self.controls)
        self.plot_comparison(self.year_slider.value, self.gender_dropdown.value)
