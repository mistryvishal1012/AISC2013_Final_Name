import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from prediction import predict
import streamlit.components.v1 as components
import time

global my_script

def validate_input(violation_code, fine_amount, penalty_amount, interest_amount,
                   payment_amount, amount_due, violation_precint, issuer_precint,
                   license_type, law_section, violation_county, issuing_agency,
                   issuing_command, violation_date, violation_time):
    
    global my_script

    if not violation_code:    
        my_script = f"<script>alert('Violation code is required.')</script>"
        return False
    
    if fine_amount == 0 :
        my_script = f"<script>alert('Fine Amount is required.')</script>"
        return False
    

    if not violation_precint:
        my_script = f"<script>alert('Violation precinct is required.')</script>"
        return False
    
    if not issuer_precint:
        my_script = f"<script>alert('Issuer precinct is required.')</script>"
        return False
    
    if not license_type:
        my_script = f"<script>alert('License type is required.')</script>"
        return False
    
    if not law_section:
        my_script = f"<script>alert('Law section is required.')</script>"
        return False
    
    if not violation_county:
        my_script = f"<script>alert('Violation county is required.')</script>"
        return False
    
    if not issuing_agency:
        my_script = f"<script>alert('Issuing agency is required.')</script>"
        return False
    
    if not issuing_command:
        my_script = f"<script>alert('Issuing command is required.')</script>"
        return False
    
    if not violation_date:
        my_script = f"<script>alert('Violation date is required.')</script>"
        return False
    
    if not violation_time:
        my_script = f"<script>alert('Violation time is required.')</script>"
        return False

    # Additional validation checks based on your requirements
    # ...

    return True

def main():  
    st.title('NYC Parking Ticket Reduction')
    st.markdown('Try Your Luck to Reduce Your Parking Ticket Fine.')

    parking_data = pd.read_csv('data/parkingdata.csv')

    Headers = pd.read_csv('data/Headers.csv')

    violation_fine = pd.read_excel('data/ParkingViolationCodes_January2020.xlsx')

    selectiondata = {
        'Violation Precinct': pd.read_csv("data/Violation Precinct.csv",header=None),
        'Issuer Precinct': pd.read_csv("data/Issuer Precinct.csv",header=None),
        'License Type': pd.read_csv("data/License Type.csv",header=None),
        'Violation': pd.read_csv("data/Violation.csv",header=None),
        'County': pd.read_csv("data/County.csv",header=None),
        'Issuing Agency': pd.read_csv("data/Issuing Agency.csv",header=None),
        'Issuer Command': pd.read_csv("data/Issuer Command.csv",header=None),
        'Law Section': pd.read_csv("data/Law Section.csv",header=None),
        'Time': pd.read_csv("data/Time.csv",header=None),
        'DayOfWeek': pd.read_csv("data/DayOfWeek.csv",header=None)
    }

    st.header("Predict")

    violation_code = st.selectbox('**Select Violation**', selectiondata['Violation'], index=None, placeholder="Select Violation", key='violation_code')
    fine_amount = st.number_input('Enter Your Fine Amount', min_value=0, key='fine_amount')
    penalty_amount = st.number_input('Enter Your Penalty Amount', min_value=0, key='penalty_amount')
    interest_amount = st.number_input('Enter Your Interest Amount', min_value=0, key='interest_amount')
    payment_amount = st.number_input('Enter Your Payment Amount', min_value=0, key='payment_amount')
    amount_due = st.number_input('Enter Your Amount Due', min_value=0, key='amount_due')
    violation_precint = st.selectbox('**Select Violation Precint**', selectiondata['Violation Precinct'], index=None, placeholder="Select Violation Precint", key='violation_precint')
    issuer_precint = st.selectbox('**Select Issuer Precint**', selectiondata['Issuer Precinct'], index=None, placeholder="Select Issuer Precint", key='issuer_precint')
    license_type = st.selectbox('**Select License Type**', selectiondata['License Type'], index=None, placeholder="Select License Type", key='license_type')
    law_section = st.selectbox('**Select Law Section**', selectiondata['Law Section'], index=None, placeholder="Select Law Section", key='law_section')
    violation_county = st.selectbox('**Select Violation County**', selectiondata['County'], index=None, placeholder="Select Violation County", key='violation_county')
    issuing_agency = st.selectbox('**Select Issuing Agency**', selectiondata['Issuing Agency'], index=None, placeholder="Select Issuing Agency", key='issuing_agency')
    issuing_command = st.selectbox('**Select Issuer Command**', selectiondata['Issuer Command'], index=None, placeholder="Select Issuer Command", key='issuing_command')
    violation_date = st.date_input("Enter Violation Date", value=None, format="YYYY/MM/DD", max_value=datetime.today(), key='violation_date')
    violation_time = st.time_input("Enter Violation Time", value=None, key='violation_time')
    st.text('')
    if st.button("Calculate"):
        # Perform form validation
        if validate_input(violation_code, fine_amount, penalty_amount, interest_amount,
                          payment_amount, amount_due, violation_precint, issuer_precint,
                          license_type, law_section, violation_county, issuing_agency,
                          issuing_command, violation_date, violation_time):
            
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            actual_fine_amount = violation_fine[violation_fine['VIOLATION DESCRIPTION'] == violation_code]

            if violation_county == 'New York':
                actual_fine_amount = actual_fine_amount["Manhattan  96th St. & below\n(Fine Amount $)"]
            else:
                actual_fine_amount = actual_fine_amount["All Other Areas\n(Fine Amount $)"]

            date_string = violation_date.strftime('%Y-%m-%d %H:%M:%S')
            date_obj = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
            day_of_week = date_obj.strftime('%A')
            violation_time = str(violation_time)
            violation_time = violation_time.split(":")[0]

            parking_data['Fine Amount'] = fine_amount
            parking_data['Penalty Amount'] = penalty_amount
            parking_data['Interest Amount'] = interest_amount
            parking_data['Payment Amount'] = payment_amount
            parking_data['Amount Due'] = amount_due
            parking_data['Violation Precinct_'+violation_precint] = 1
            parking_data['Issuer Precinct_'+issuer_precint] = 1

            if int(fine_amount) > int(actual_fine_amount):
                parking_data['Is_High'] = True
            else:
                parking_data['Is_High'] = False

            parking_data['License Type_' + license_type] = 1
            parking_data['Violation_' + violation_code] = 1
            parking_data['County_' + violation_county] = 1
            parking_data['Issuing Agency_' + issuing_agency] = 1
            parking_data['Issuer Command_' + issuing_command] = 1
            parking_data['Law Section_' + law_section] = 1
            parking_data['Time_' + violation_time + ':00'] = 1
            parking_data['DayOfWeek_' + day_of_week] = 1

            with st.spinner("Calculating"):
                result = predict(parking_data)
                st.text(f'Your Parking Ticket can be Reduced by {result[0]} %')
                
        else :
            components.html(f"{my_script}")
              

if __name__ == "__main__":
    main()
