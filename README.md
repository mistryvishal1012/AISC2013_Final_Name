
# NYC Parking Ticket Reduction With Docker

## File Structure :
- data
    - Violation Precinct.csv    
    - Issuer Precinct.csv
    - License Type.csv
    - Violation.csv
    - County.csv
    - Issuing Agency.csv
    - Issuer Command.csv
    - Law Section.csv
    - Time.csv
    - DayOfWeek.csv
- DockerFile 
- app.py
- best.sav.gz
- prediction.py
- requirements.txt

## Requiremets :
1) joblib
2) streamlit
3) pandas
4) numpy
5) scikit-learn
6) openpyxl

## Run Docker File

### Build a Docker Image - 
docker build -t docker-image-name . 

### Run Docker Image at Port - 
docker run -p port:port docker-image-name
