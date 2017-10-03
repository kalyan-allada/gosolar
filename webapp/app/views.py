from flask import render_template, url_for
from app import app
from flask import request
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import pickle as pk
from uszipcode import ZipcodeSearchEngine

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/output')
def output():
    
    array_type = str(request.args.get('array_type'))
    tracking_type = str(request.args.get('tracking_type'))
    company = str(request.args.get('company'))

    
    array_size = request.args.get('array_size')
    try:    
        size_num = float(array_size) 
    except ValueError:  
        return render_template("value_error.html" )
    else:  
        # limit size to 100 kW whichs is equal to 625 sq.m
        if ((size_num > 0.0) and (size_num <= 625.0)):
            pass
        else:   
            return render_template("value_error.html")
    
    
    zipcode = str(request.args.get('zipcode'))
    try:    
        zip_num = int(zipcode) 
    except ValueError:
        return render_template("value_error.html" )
    else:  
        #check for MA zipcodes
        if ((zip_num >= 1001) and (zip_num <= 2790)):
            pass
        else:   
            return render_template("value_error.html")
    
    
    tilt_angle = request.args.get('tilt_angle')
    try:    
        tilt_num = float(tilt_angle) 
    except ValueError:
        return render_template("value_error.html" )
    else:  
        if ((tilt_num >= 0) and (tilt_num <= 90)):
            pass
        else:   
            return render_template("value_error.html")
    
    
    # Convert array_size units to kW
    # 1kW = 1 m^2 * 1 kW/m^2 * 0.16 , 0.16 is the mean efficiency (NREL pvwatts)
    array_size_kw = float(array_size) * 0.16
    
    
    # Use average..there's not much dependence on this value
    annual_insolation = 4.288 
    
    tech_dict = {'poly': 1, 'mono': 2, 'cryst': 3, 'mono_asi':4, 'asi':5,\
             'multiple':6}
    track_dict = {'fixed': 1, 'single-axis': 2, 'dual-axis': 3}
    
    company_dict = {'c20': 90, 'c11': 274, 'c15': 124, 'c17': 110, 'c19': 95,\
                    'c18': 104, 'c16': 118, 'c13': 199, 'c9': 284, 'c8': 332,\
                    'c14': 184, 'c12': 233, 'c3': 874, 'c2': 2356, 'c1': 2989,\
                    'c10': 278, 'c7': 405, 'c6': 458, 'c5': 557, 'c4': 855}
    
    array_type = pd.Series(array_type)
    tracking_type = pd.Series(tracking_type)
    company = pd.Series(company)
     
    array_type = array_type.map(tech_dict).astype(int)
    tracking_type = tracking_type.map(track_dict).astype(int)
    company = company.map(company_dict).astype(int)
    
    
    #Convert zipcode to lat, lon before feeding to the model
    search = ZipcodeSearchEngine()
    zip_info = search.by_zipcode(zipcode)

    lat = zip_info.Latitude
    lon = zip_info.Longitude
                   
    user_data_cost = [[float(array_size_kw), float(company[0]), float(array_type[0]),\
                       float(tracking_type[0]), lat, lon]]
    
    user_data_energy = [[float(array_size_kw), annual_insolation, float(tilt_angle),\
                         float(tracking_type[0]), float(array_type[0]), lat, lon]]
    
    print user_data_cost
    print user_data_energy
    
    predictions = []
    
    pred_cost  = pk.load(open('./data/cost_ma_final.pk','rb'))
    pred_energy  = pk.load(open('./data/energy_ma_final.pk','rb'))
    
    install_cost = round(pred_cost.predict(user_data_cost))
    predictions.append([install_cost])
    
    annual_energy =  round(pred_energy.predict(user_data_energy), 2)
    predictions.append([annual_energy])
    
    # Avg cost per kwh in MA = 0.1983
    annual_return = round(annual_energy * 0.1983, 2)
    predictions.append([annual_return])

    #MA avg incentive is 2K
    # Govt. incentives can vary from 30 - 65% of the total cost  
    # num of year for break even
    num_years_low = round(((install_cost - 0.30*install_cost)/annual_return), 1)
    predictions.append([num_years_low])
    
    num_years_high = round(((install_cost - 0.65*install_cost)/annual_return), 1)
    predictions.append([num_years_high])
    
    #print predictions
    
    the_result = ''
    return render_template("output.html",  predictions = predictions, \
                           the_result = the_result)
