from flask import Flask, render_template, url_for, request, session, redirect, flash
import pymongo
import bcrypt
from bson.objectid import ObjectId
from functools import wraps
import pickle
import pandas as pd
import numpy as np
# from flask_login import login_required,LoginManager
# login_manager = LoginManager()

from werkzeug.utils import secure_filename
import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STAT_DIR = os.path.join(APP_ROOT,'static')
 
app = Flask(__name__)
# login_manager.init_app(app)

 
app.secret_key = 'super secret key'
myclient = pymongo.MongoClient("mongodb://localhost:27017/")

###### admin database #######
mydb = myclient["addb"]
records = mydb["user"]
medi_table = mydb["medicine"]
doctor_table = mydb["doctor"]
tips_table = mydb["tips"]

####### normal database #####

db = myclient["nordb"]
user_table = db["userpat"]
contact_table = db["contact"]
appoint_table = db["appointment"]
fav_tips = db["favtips"]
fav_medicine = db["favmedi"]
feedback_table = db["feedback"]



 
@app.route('/', methods=['GET', "POST"])
def home():
    return render_template("index2.html", **locals())


@app.route('/contact', methods=['GET', "POST"])
def contact():
    if request.method == "POST":
        contact_table.insert_one(dict(request.form))
    return render_template("contact2.html", **locals())

####### model building  #################################
model = pickle.load(open("Diabetes3.pkl", "rb"))
model1 = pickle.load(open("stroke.pkl", "rb"))
model2 = pickle.load(open("cancer.pkl", "rb"))
model3 = pickle.load(open("heart.pkl", "rb"))

@app.route('/heart', methods=['GET','POST'])
def heart():
    return render_template("heart.html")
@app.route('/diebetic', methods=['GET','POST'])
def diebetic():
    return render_template("diebetic.html")
@app.route('/stroke', methods=['GET','POST'])
def stroke():
    return render_template("stroke.html")
@app.route('/cancer', methods=['GET','POST'])
def cancer():
    return render_template("cancer.html")



@app.route('/predict1',methods=['POST','GET'])
def predict1():
    text1 = int(request.form['1'])
    text2 = int(request.form['2'])
    text3 = int(request.form['3'])
    text4 = int(request.form['4'])
    text5 = int(request.form['5'])
    text6 = int(request.form['6'])
    text7 = int(request.form['7'])
    text8 = int(request.form['8'])
    text9 = int(request.form['9'])
    text10 = int(request.form['10'])

 
    data = np.array([[text1,text2,text3,text4,text5,text6,text7,text8,text9,text10]])
    my_prediction = model1.predict(data)
        
    return render_template('stroke_result.html', prediction=my_prediction)

@app.route('/predict2',methods=['POST','GET'])
def predict2():
    text1 = int(request.form['1'])
    text2 = int(request.form['2'])
    text3 = int(request.form['3'])
    text4 = int(request.form['4'])
    text5 = int(request.form['5'])
    text6 = int(request.form['6'])
    text7 = int(request.form['7'])
    text8 = int(request.form['8'])
    text9 = int(request.form['9'])
    text10 = int(request.form['10'])
    text11 = int(request.form['11'])
    text12 = int(request.form['12'])
    text13 = int(request.form['13'])
    text14 = int(request.form['14'])
    text15 = int(request.form['15'])
    text14 = int(request.form['14'])
    text15 = int(request.form['15'])
    text16 = int(request.form['16'])
 
    row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8,text9,text10,text11,text12,text13,text14,text15,text16])])
    print(row_df)
    prediction=model.predict_proba(row_df)
    output='{0:.{1}f}'.format(prediction[0][1], 2)
    output = str(float(output)*100)+'%'
    if output>str(0.5):
        return render_template('die_result.html',pred=f'You have chance of having diebetics.\nProbability of having diebetics is {output}')
    else:
        return render_template('die_result.html',pred=f'You are safe.\n Probability of having diebetics is {output}')    

@app.route('/predict3',methods=['POST','GET'])
def predict3():
    text1 = int(request.form['1'])
    text2 = int(request.form['2'])
    text3 = int(request.form['3'])
    text4 = int(request.form['4'])
    text5 = int(request.form['5'])
    text6 = int(request.form['6'])
    text7 = int(request.form['7'])
    text8 = int(request.form['8'])
    text9 = int(request.form['9'])
    text10 = int(request.form['10'])
    text11 = int(request.form['11'])
    text12 = int(request.form['12'])
    text13 = int(request.form['13'])


    data = np.array([[text1,text2,text3,text4,text5,text6,text7,text8,text9,text10,text11,text12,text13]])
    my_prediction = model3.predict(data)
        
    return render_template('heart_result.html', prediction=my_prediction) 

@app.route('/predict4',methods=['POST','GET'])
def predict4():
    text1 = int(request.form['1'])
    text2 = int(request.form['2'])
    text3 = int(request.form['3'])
    text4 = int(request.form['4'])
    text5 = int(request.form['5'])
    text6 = int(request.form['6'])
    text7 = int(request.form['7'])
    text8 = int(request.form['8'])
    text9 = int(request.form['9'])
    text10 = int(request.form['10'])
    text11 = int(request.form['11'])
    text12 = int(request.form['12'])
    text13 = int(request.form['13'])
    text14 = int(request.form['14'])
    text15 = int(request.form['15'])


    data = np.array([[text1,text2,text3,text4,text5,text6,text7,text8,text9,text10,text11,text12,text13,text14,text15]])
    my_prediction = model2.predict(data)
        
    return render_template('cancer_result.html', prediction=my_prediction)



######### authentication for admin #########

@app.route("/signup", methods=['post', 'get'])
def signup():
    message = ''
    if "email" in session:
        return redirect(url_for("logged_in"))
    if request.method == "POST":
        user = request.form.get("name")
        email = request.form.get("email")
        password1 = request.form.get("password1")
        password2 = request.form.get("password2")
        user_found = records.find_one({"name": user})
        email_found = records.find_one({"email": email})
        if user_found:
            message = 'There already is a user by that name'
            return render_template('signup.html', **locals())
        if email_found:
            message = 'This email already exists in database'
            return render_template('signup.html', **locals())
        if password1 != password2:
            message = 'Passwords should match!'
            return render_template('signup.html', **locals())
        else:
            hashed = bcrypt.hashpw(password2.encode('utf-8'), bcrypt.gensalt()) # hashing
            user_input = {'name': user, 'email': email, 'password': hashed} #storing in dictionary
            records.insert_one(user_input)
            user_data = records.find_one({"email": email})
            new_email = user_data['email']
            return redirect(url_for("login"))
    return render_template('signup.html')

@app.route("/login", methods=["POST", "GET"])
def login():
    message = ''
    if "email" in session:
        return redirect(url_for("logged_in"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        email_found = records.find_one({"email": email})
        if email_found:
            email_val = email_found['email']
            passwordcheck = email_found['password']
            if bcrypt.checkpw(password.encode('utf-8'), passwordcheck):
                session['logged_in'] = True
                session["email"] = email_val
                return redirect(url_for('logged_in'))
            else:
                if "email" in session:
                    return redirect(url_for("logged_in"))
                message = 'Wrong password'
                return render_template('login.html', **locals())
        else:
            message = 'Email not found'
            return render_template('login.html', **locals())
    return render_template('login.html', **locals())

@app.route('/logged_in')
def logged_in():
    if "email" in session:
        email = session["email"]
        return render_template('logged_in.html', **locals())
    else:
        return redirect(url_for("login"))

@app.route("/logout", methods=["POST", "GET"])
def logout():
    session.clear()
    return render_template('index2.html')

def login_required(f):
  @wraps(f)
  def wrap(*args, **kwargs):
    if 'logged_in' in session:
      return f(*args, **kwargs)
    else:
      return "<h1> not authorized </h1>"
  return wrap

####### authentication for patient #######

@app.route("/signup_pat", methods=['post', 'get'])
def signup_pat():
    message = ''
    if "email_pat" in session:
        return redirect(url_for("logged_in_pat"))
    if request.method == "POST":
        user = request.form.get("name")
        email = request.form.get("email")
        password1 = request.form.get("password1")
        password2 = request.form.get("password2")
        user_found = user_table.find_one({"name": user})
        email_found = user_table.find_one({"email": email})
        if user_found:
            message = 'There already is a user by that name'
            return render_template('signup_pat.html', **locals())
        if email_found:
            message = 'This email already exists in database'
            return render_template('signup_pat.html', **locals())
        if password1 != password2:
            message = 'Passwords should match!'
            return render_template('signup_pat.html', **locals())
        else:
            hashed = bcrypt.hashpw(password2.encode('utf-8'), bcrypt.gensalt()) # hashing
            user_input = {'name': user, 'email': email, 'password': hashed} #storing in dictionary
            user_table.insert_one(user_input)
            user_data = user_table.find_one({"email": email})
            new_email = user_data['email']
            return redirect(url_for("login_pat"))
    return render_template('signup_pat.html')

@app.route("/login_pat", methods=["POST", "GET"])
def login_pat():
    message = ''
    if "email_pat" in session:
        return redirect(url_for("logged_in_pat"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        email_found = user_table.find_one({"email": email})
        if email_found:
            email_val = email_found['email']
            passwordcheck = email_found['password']
            if bcrypt.checkpw(password.encode('utf-8'), passwordcheck):
                session['logged_in_pat'] = True
                session["email_pat"] = email_val
                return redirect(url_for('logged_in_pat'))
            else:
                if "email_pat" in session:
                    return redirect(url_for("logged_in_pat"))
                message = 'Wrong password'
                return render_template('login_pat.html', **locals())
        else:
            message = 'Email not found'
            return render_template('login_pat.html', **locals())
    return render_template('login_pat.html', **locals())

@app.route('/logged_in_pat')
def logged_in_pat():
    if "email_pat" in session:
        email = session["email_pat"]
        return render_template('logged_in_pat.html', **locals())
    else:
        return redirect(url_for("login_pat"))


def login_required_pat(f):
  @wraps(f)
  def wrap(*args, **kwargs):
    if 'logged_in_pat' in session:
      return f(*args, **kwargs)
    else:
      return "<h1> not authorized </h1> "
  return wrap

######### show contact, appointment ##############

@app.route('/showcontact', methods=['GET', "POST"])
def showcontact():
    contacts = contact_table.find()
    return render_template("showcontact.html", **locals())



######### show doctor, medicine, tips for visitor users ##############

@app.route('/norshowdoctor', methods=['GET', "POST"])
def norshowdoctor():
    doctors = doctor_table.find()
    return render_template("norshowdoctor.html", **locals())

@app.route('/norshowmedicine', methods=['GET', "POST"])
def norshowmedicine():
    medicines = medi_table.find()
    return render_template("norshowmedicine.html", **locals())

@app.route('/norshowtips', methods=['GET', "POST"])
def norshowtips():
    tips = tips_table.find()
    return render_template("norshowtips.html", **locals())

######### show doctor, medicine, tips and other functionality for patient users ##############

@app.route('/patshowdoctor', methods=['GET', "POST"])
@login_required_pat
def patshowdoctor():
    doctors = doctor_table.find()
    if request.method == "POST":  
        key = request.form["doctor"]
        print(key)
        docs = doctor_table.find({'speciality':{'$regex':key,"$options":"i"}})
        return render_template("searchdoctor.html", **locals())
    return render_template("patshowdoctor.html", **locals())

@app.route('/patshowmedicine', methods=['GET', "POST"])
@login_required_pat
def patshowmedicine():
    medicines = medi_table.find()
    if request.method == "POST":  
        key = request.form["medicine"]
        medicines = medi_table.find({'name':{'$regex':key,"$options":"i"}})
        return render_template("searchmedicine.html", **locals())
    return render_template("patshowmedicine.html", **locals())

@app.route('/patshowtips', methods=['GET', "POST"])
@login_required_pat
def patshowtips():
    tips = tips_table.find()
    if request.method == "POST":  
        key = request.form["tips"]
        tips = tips_table.find({'title':{'$regex':key,"$options":"i"}})
        return render_template("searchtips.html", **locals())
    return render_template("patshowtips.html", **locals())

@app.route('/appointment', methods=['GET', "POST"])
@login_required_pat
def appointment():
    doctors = doctor_table.find()
    valemail = session["email_pat"]
    if request.method == "POST":
        email = request.form["email"]
        contact = request.form["contact"]
        name = request.form["name"]
        typei = request.form["type"]
        date = request.form["date"]
        age= request.form["age"]
        desc = request.form["desc"]

        appoint_table.insert_one({"valid":valemail,"email": email,"contact": contact,"name": name,"typei": typei,"date": date,"age": age,"desc": desc})
    return render_template("appointment.html", **locals())

### for admin ####
@app.route('/showappoint', methods=['GET', "POST"])
@login_required
def showappoint():
    appoints = appoint_table.find()
    return render_template("showappoint.html", **locals())

@app.route('/<id>/feedback', methods=['GET', "POST"])
def feedback(id):
    appoint = appoint_table.find_one({'_id':ObjectId(id) })
    if request.method == 'POST':
        title = request.form["title"]
        body = request.form["body"]
        feedback_table.insert_one({"title": title,"body": body,"valmail":appoint["valid"]})

    return render_template("feedback.html", **locals())

@app.route('/getfeedback', methods=['GET', "POST"])
@login_required_pat
def getfeedback():
    email = session["email_pat"] 
    feeds = feedback_table.find({"valmail": email})

    return render_template("getfeedback.html", **locals())

################## favourite tips #################
@app.route('/<id>/favtips', methods=['GET', "POST"])
def favtips(id): 
    tips = tips_table.find({'_id':ObjectId(id) })
    ti = tips_table.find_one({'_id':ObjectId(id) })
    email = session["email_pat"]
    fa = fav_tips.insert_one({"title":ti["title"],"body":ti["body"],"email":email}) # this is storing a users favourite tips

    return render_template("favtips.html", **locals())

@app.route('/favtipslist', methods=['GET', "POST"])
@login_required_pat
def favtipslist():
        
    email = session["email_pat"] 
    tips = fav_tips.find({"email": email})
    
    return render_template("favtipslist.html", **locals())

@app.route('/<id>/deletefavdoctor/', methods=['GET', "POST"])
@login_required_pat
def deletefavtips(id):
    fav_tips.delete_many({"_id": ObjectId(id)})
    return redirect("/favtipslist")

######## favourte medicine #####

@app.route('/<id>/favmedi', methods=['GET', "POST"])
def favmedi(id): 
    tips = medi_table.find({'_id':ObjectId(id) })
    ti = medi_table.find_one({'_id':ObjectId(id) })
    email = session["email_pat"]
    fa = fav_medicine.insert_one({"name":ti["name"],"type":ti["disease_type"],"desc":ti["desc"],"email":email}) # this is storing a users favourite tips

    return render_template("favmedi.html", **locals())

@app.route('/favmedilist', methods=['GET', "POST"])
@login_required_pat
def favmedilist():
        
    email = session["email_pat"] 
    tips = fav_medicine.find({"email": email})
    
    return render_template("favmedilist.html", **locals())

@app.route('/<id>/deletefavmedi/', methods=['GET', "POST"])
@login_required_pat
def deletefavmedi(id):
    fav_medicine.delete_many({"_id": ObjectId(id)})
    return redirect("/favmedilist")

# @app.route('/patshowdoctor', methods=['GET', "POST"])
# def searchdoctor():
#     if request.method == "POST":  
#         key = request.form["doctor"]
#         print(key)
#         doctors = doctor_table.find({"speciality": key})

#         return render_template("searchdoctor.html", **locals())
        
#     return render_template("patshowdoctor.html", **locals())

# @app.route('/patshowmedicine', methods=['GET', "POST"])
# def searchmedicine():
#     if request.method == "POST":  
#         key = request.form["medicine"]
#         medicines = medi_table.find({"name": key})
#         return render_template("searchmedicine.html", **locals())
        
#     return render_template("patshowmedicine.html", **locals())

# @app.route('/patshowtips', methods=['GET', "POST"])
# def searchtips():
#     if request.method == "POST":  
#         key = request.form["tips"]
#         tips = tips_table.find({"title": key})
#         return render_template("searchtips.html", **locals())
        
#     return render_template("patshowtips.html", **locals())

###### functions for admin ######
####################### doctor #######################

@app.route('/createdoctor', methods=['GET', "POST"])
@login_required
def createdoctor():
    if request.method == "POST":
        doctor_table.insert_one(dict(request.form))
        return redirect("/showdoctor")

    return render_template("createdoctor.html", **locals())

@app.route('/<id>/editdoctor', methods=['GET', "POST"])
@login_required
def editdoctor(id):
    doctors = doctor_table.find_one({'_id':ObjectId(id) })
    if request.method == "POST":   
        form_data = dict(request.form)
        email = form_data["email"]
        contact = form_data["contact"]
        name = form_data["name"]
        speciality = form_data["speciality"]
        institution = form_data["institution"]
        doctor_table.update_many({'_id':ObjectId(id) },{ "$set": {"email": email,"contact": contact,"name": name,"speciality": speciality,"institution": institution} })
        return redirect("/showdoctor")
        
    return render_template("editdoctor.html", **locals())


@app.route('/<id>/deletedoctor/', methods=['GET', "POST"])
@login_required
def deletedoctor(id):
    doctor_table.delete_many({"_id": ObjectId(id)})
    return redirect("/showdoctor")

@app.route('/showdoctor', methods=['GET', "POST"])
@login_required
def showdoctor():
    doctors = doctor_table.find()
    return render_template("showdoctor.html", **locals())

################################ medicine ##############################################

@app.route('/createmedicine', methods=['GET', "POST"])
@login_required
def createmedicine():
    target = os.path.join(STAT_DIR, 'uploads/')  #folder path
    if not os.path.isdir(target):
            os.mkdir(target)     
    if request.method == 'POST':
        name = request.form["name"]
        disease_type = request.form["disease_type"]
        desc = request.form["desc"]
        for upload in request.files.getlist("image"): #multiple image handel
            filename = secure_filename(upload.filename)
            destination = "/".join([target, filename])
            upload.save(destination)
            medi_table.insert_one({'name':name,'disease_type':disease_type, 'desc':desc, 'image': filename}) 

    return render_template("createmedicine.html", **locals())
    
@app.route('/<id>/editmedicine', methods=['GET', "POST"])
@login_required
def editmedicine(id):
    medi = medi_table.find_one({'_id':ObjectId(id) })
    target = os.path.join(STAT_DIR, 'uploads/')  #folder path
    if not os.path.isdir(target):
            os.mkdir(target)     
    if request.method == 'POST':
        name = request.form["name"]
        disease_type = request.form["disease_type"]
        desc = request.form["desc"]
        for upload in request.files.getlist("image"): #multiple image handel
            filename = secure_filename(upload.filename)
            destination = "/".join([target, filename])
            upload.save(destination)
            medi_table.update_many({'_id':ObjectId(id) },{ "$set": {'name':name,'disease_type':disease_type, 'desc':desc, 'image': filename}}) 
        
    return render_template("editmedicine.html", **locals())

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/<id>/deletemedicine/', methods=['GET', "POST"])
def deletemedicine(id):
    medi_table.delete_many({"_id": ObjectId(id)})
    return redirect("/showmedicine")

@app.route('/showmedicine', methods=['GET', "POST"])
@login_required
def showmedicine():
    medicines = medi_table.find()
    return render_template("showmedicine.html", **locals())

################################ tips ##############################################

@app.route('/createtips', methods=['GET', "POST"])
@login_required
def createtips():
    if request.method == "POST":
        tips_table.insert_one(dict(request.form))
        return redirect("/showtips")

    return render_template("createtips.html", **locals())

@app.route('/<id>/edittips', methods=['GET', "POST"])
@login_required
def edittips(id):
    tips = tips_table.find_one({'_id':ObjectId(id) })
    if request.method == "POST":   
        form_data = dict(request.form)
        form_title = form_data["title"]
        form_body = form_data["body"]
        tips_table.update_many({'_id':ObjectId(id) },{ "$set": {"title": form_title,"body": form_body} })
        return redirect("/showtips")
        
    return render_template("edittips.html", **locals())


@app.route('/<id>/deletetips/', methods=['GET', "POST"])
@login_required
def deletetips(id):
    tips_table.delete_many({"_id": ObjectId(id)})
    return redirect("/showtips")

@app.route('/showtips', methods=['GET', "POST"])
@login_required
def showtips():
    tips = tips_table.find()
    return render_template("showtips.html", **locals())

@app.route('/contactpat', methods=['GET', "POST"])
@login_required_pat
def contactpat():
    if request.method == "POST":
        contact_table.insert_one(dict(request.form))
    return render_template("contactpat.html", **locals())


if __name__ == "__main__":
    app.run(debug=True)
    app.run()