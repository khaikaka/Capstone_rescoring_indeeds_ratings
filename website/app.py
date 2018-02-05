from __future__ import division
from flask import Flask, render_template, request, jsonify
from math import sqrt
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_data = request.json
    array = np.array([[ 0.46135632,  0.41142069,  0.26545529,  0.22678085,  0.42187403], \
           [ 0.31445019,  0.35496115,  0.16028048,  0.19169524,  0.28681548], \
           [ 0.28030069,  0.20269881,  0.16942706,  0.10643459,  0.30486937], \
           [ 0.30053908,  0.36147373,  0.26920781,  0.20980763,  0.4105872 ], \
           [ 0.34162824,  0.34192006,  0.18877151,  0.114261  ,  0.2431674 ], \
           [ 0.31261959,  0.33319636,  0.17501111,  0.18717503,  0.31028023], \
           [ 0.26811249,  0.28810182,  0.15769132,  0.13622988,  0.27839978], \
           [ 0.32379   ,  0.24869656,  0.23322272, -0.01376181,  0.33706041], \
           [ 0.33891147,  0.3416846 ,  0.20204019,  0.18280032,  0.37477233], \
           [ 0.23419666,  0.18858931,  0.10166337,  0.17809265,  0.22842272], \
           [ 0.30344594,  0.38830619,  0.22295761,  0.15276601,  0.2862725 ], \
           [ 0.23362989,  0.2164504 ,  0.14302241,  0.05971799,  0.27852177], \
           [ 0.1556197 ,  0.24094586,  0.23328322,  0.12630354,  0.24958661], \
           [ 0.35474386,  0.43352838,  0.33324503,  0.17722421,  0.35303585], \
           [ 0.26018261,  0.35885551,  0.20253548,  0.12559103,  0.30179112], \
           [ 0.29288196,  0.3410968 ,  0.15039979,  0.15673993,  0.28752514], \
           [ 0.23204301,  0.37563892,  0.24759373,  0.09073805,  0.14864976], \
           [ 0.25531972,  0.26433648,  0.25915977,  0.12860266,  0.33424437], \
           [ 0.27633945,  0.19625124,  0.11874841,  0.07206441,  0.18995087], \
           [ 0.27502177,  0.30533108,  0.16034749,  0.04781765,  0.29036297], \
           [-0.14864648, -0.03492425,  0.12873654, -0.23138951, -0.23277649], \
           [ 0.38148626,  0.4074692 ,  0.31767003,  0.1435732 ,  0.44661127], \
           [ 0.42538918,  0.38289236,  0.22230289,  0.14706224,  0.31496077], \
           [ 0.27751371,  0.05441099,  0.23238868,  0.23840903,  0.39123631], \
           [ 0.03789574,  0.15809499,  0.17800038, -0.0728249 ,  0.177432  ], \
           [ 0.32294033,  0.3584506 ,  0.17683467,  0.13237404,  0.22870641], \
           [ 0.15592   ,  0.25272914,  0.03258111,  0.01827491,  0.40359464], \
           [ 0.38671096,  0.07420808, -0.15574345, -0.15319367,  0.37998713], \
           [ 0.42075549,  0.35832419,  0.24550041,  0.06670094,  0.22950854], \
           [ 0.32459583,  0.40306788,  0.26996227,  0.12864383,  0.39669461], \
           [ 0.2726697 ,  0.30974949,  0.17810545,  0.08832228,  0.35751932]])
    company = ['Adobe','Airbnb','Allstate','Apple','Boeing','Cisco','Dell', 'Expedia', \
               'Google','IBM','Intel','JLL','KPMG','Kaiser Permanente',      \
               'Microsoft','NOKIA','Netflix','Nordstrom','Oracle','Qualcomm','Redfin', \
               'Salesforce','T-mobile','Tableau','Tesla','Texas Instrument','Twitter', \
               'Uber','University of Washington','Workday','Zillow']
    wlb, pbt, jsa, mng, cul = int(user_data['wlb']), int(user_data['pbt']), int(user_data['jsa']), int(user_data['mng']), int(user_data['cul'])
    list_companies =  _recommender_sytem(array, company, wlb, pbt, jsa, mng, cul)
    return render_template("companies_table.html",
                           companies=list_companies,
                           company_numbers=[i for i in range(1,len(list_companies)+1)])



def _recommender_sytem(array, company, wlb, pbt, jsa, mng, cul):
    array_transpose = array.transpose()
    company_array = np.array(company)
    user_array = np.array([wlb,pbt,jsa,mng, cul])
    user = np.dot(user_array, array_transpose)
    rank = np.argsort(user)[::-1]
    return company_array[rank][:10]


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
