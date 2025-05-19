from flask import Flask,render_template, request, redirect
import util


app=Flask(__name__)
@app.route('/')
def home():
    return render_template("home.html")


@app.route("/verify",methods=["POST"])
def verify():
    image_data=request.files.get("image")
    if image_data:
        prediction=util.predict(image_data)
        if prediction==0:
            return render_template("home.html",result="You dont have retinitis pigmentosa")
        else:
            return render_template("home.html",result="You have retinitis Pigmentosa")
    else:
        return redirect("/")

if __name__=="__main__":
    app.run(debug=True)
