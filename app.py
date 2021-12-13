from flask import Flask, render_template, request
import repository
import joblib
import pandas

app = Flask(__name__)


def do_predict(
    City,
    Zip_Code,
    Gender,
    Senior_Citizen,
    Partner,
    Dependents,
    Tenure_Months,
    Phone_Service,
    Multiple_Lines,
    Internet_Service,
    Online_Security,
    Online_Backup,
    Device_Protection,
    Tech_Support,
    Streaming_TV,
    Streaming_Movies,
    Contract,
    Paperless_Billing,
    Payment_Method,
    Monthly_Charges,
    Total_Charges,
):
    predict = {
        "City": City,
        "Zip Code": Zip_Code,
        "Gender": Gender,
        "Senior Citizen": Senior_Citizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "Tenure Months": Tenure_Months,
        "Phone Service": Phone_Service,
        "Multiple Lines": Multiple_Lines,
        "Internet Service": Internet_Service,
        "Online Security": Online_Security,
        "Online Backup": Online_Backup,
        "Device Protection": Device_Protection,
        "Tech Support": Tech_Support,
        "Streaming TV": Streaming_TV,
        "Streaming Movies": Streaming_Movies,
        "Contract": Contract,
        "Paperless Billing": Paperless_Billing,
        "Payment Method": Payment_Method,
        "Monthly Charges": Monthly_Charges,
        "Total Charges": Total_Charges,
    }

    to_data_frame = pandas.DataFrame(data=predict, index=[0])

    the_model = joblib.load("clfFinalModel4.sav")

    result = the_model.predict(to_data_frame)

    return result


@app.route("/", methods=["GET", "POST"])
def index():
    cities = repository.cities
    city_number = repository.city_number
    zip_codes = repository.zip_codes
    zip_code_number = repository.zip_code_number
    genders = repository.genders
    yes_or_no_only = repository.yes_or_no_only
    yes_no_services = repository.yes_no_services
    yes_no_services_number = repository.yes_no_services_number
    tenure_months = repository.tenure_months
    monthly_charges = repository.monthly_charges
    total_charges = repository.total_charges
    yes_or_no_only_number = repository.yes_or_no_only_number
    gender_number = repository.gender_number

    City = request.args.get("city")
    Zip_Code = request.args.get("zip-code")
    Gender = request.args.get("gender")
    Senior_Citizen = request.args.get("senior-citized")
    Partner = request.args.get("partner")
    Dependents = request.args.get("dependents")
    Tenure_Months = request.args.get("tenure-months")
    Phone_Service = request.args.get("phone-service")
    Multiple_Lines = request.args.get("multiple-line")
    Internet_Service = request.args.get("internet-service")
    Online_Security = request.args.get("online-security")
    Online_Backup = request.args.get("online-backup")
    Device_Protection = request.args.get("device-protection")
    Tech_Support = request.args.get("tech-support")
    Streaming_TV = request.args.get("streaming-tv")
    Streaming_Movies = request.args.get("streaming-movie")
    Contract = request.args.get("contract")
    Paperless_Billing = request.args.get("paperles-billing")
    Payment_Method = request.args.get("payment-method")
    Monthly_Charges = request.args.get("monthly-charges")
    Total_Charges = request.args.get("total-charges")

    result = do_predict(
        City,
        Zip_Code,
        Gender,
        Senior_Citizen,
        Partner,
        Dependents,
        Tenure_Months,
        Phone_Service,
        Multiple_Lines,
        Internet_Service,
        Online_Security,
        Online_Backup,
        Device_Protection,
        Tech_Support,
        Streaming_TV,
        Streaming_Movies,
        Contract,
        Paperless_Billing,
        Payment_Method,
        Monthly_Charges,
        Total_Charges,
    )

    return render_template(
        "index.html",
        cities=cities,
        zip_codes=zip_codes,
        genders=genders,
        yes_or_no_only=yes_or_no_only,
        tenure_months=tenure_months,
        monthly_charges=monthly_charges,
        total_charges=total_charges,
        city_number=city_number,
        zip_code_number=zip_code_number,
        yes_or_no_only_number=yes_or_no_only_number,
        gender_number=gender_number,
        yes_no_services=yes_no_services,
        yes_no_services_number=yes_no_services_number,
        result=result,
    )


if __name__ == "__main__":
    app.run(debug=True)
