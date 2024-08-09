from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import pickle
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

type_expected = [".csv", ".pdf"]


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.route('/', methods=['GET', "POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
        for ftype in type_expected:
            if ftype in file.filename:
                reg = pickle.load(open("IA_data/linearRegressionConcreteAtt.sav", 'rb'))
                p_ds = pd.read_csv(f"static/files/{file.filename}")
                resp = reg.predict(p_ds)
                return f"Arquivo enviado com sucesso!<br>Os Reusltados são:<br>Força compressiva do concreto: {resp}"

        return f"O arquivo enviado não é do tipo esperado.<br>Os tipos esperados são: {type_expected}"

    return render_template('index.html', form=form)


@app.route('/resultado_por_texto', methods=['GET', "POST"])
def resultado_por_texto():
    val_cimento = float(request.form.get("Cimento_txt"))
    val_add = float(request.form.get("Adicoes_txt"))
    val_ac = float(request.form.get("ac_txt"))
    val_af = float(request.form.get("af_txt"))
    val_ag = float(request.form.get("ag_txt"))
    val_id = int(request.form.get("id_txt"))

    reg = pickle.load(open("IA_data/linearRegressionConcreteAtt.sav", 'rb'))
    resp = reg.predict([[val_cimento, val_add, val_ac, val_af, val_ag, val_id]])
    return f"Os Reusltados são:<br>Força compressiva do concreto: {resp}"


if __name__ == '__main__':
    app.run(debug=True)
