from pydantic import BaseModel,Field,computed_field,field_validator
from typing import Literal,Annotated

# pydantic model to handle incoming data
class UserInput(BaseModel):
    A1_Score: Annotated[Literal[0,1],Field(...,description="give the A1_score for autism")]
    A2_Score:Annotated[Literal[0,1],Field(...,description="give the A2_score for autism")]
    A3_Score: Annotated[Literal[0,1],Field(...,description="give the A3_score for autism")]
    A4_Score : Annotated[Literal[0,1],Field(...,description="give the A4_score for autism")]
    A5_Score : Annotated[Literal[0,1],Field(...,description="give the A5_score for autism")]
    A6_Score:Annotated[Literal[0,1],Field(...,description="give the A6_score for autism")]
    A7_Score:Annotated[Literal[0,1],Field(...,description="give the A7_score for autism")]
    A8_Score :Annotated[Literal[0,1],Field(...,description="give the A8_score for autism")]
    A9_Score :Annotated[Literal[0,1],Field(...,description="give the A9_score for autism")]
    A10_Score : Annotated[Literal[0,1],Field(...,description="give the A10_score for autism")]
    age : Annotated[int,Field(...,description="give the age of the patient",gt=0,lt=120)]
    gender: Annotated[Literal['f','m'],Field(...,description="what is the gender of the patient")]
    ethnicity : Annotated[Literal['Others', 'White-European', 'Middle Eastern ', 'Pasifika', 'Black',
       'Hispanic', 'Asian', 'Turkish', 'South Asian', 'Latino'],Field(...,description="What is the patient's ethinicity")]
    jaundice :Annotated[Literal['yes','no'],Field(...,description="is the patient have jaundice")]
    austim:Annotated[Literal['yes','no'],Field(...,description="does the patient have any previous signs of autism")]
    contry_of_res : Annotated[Literal['Austria', 'India', 'United States', 'South Africa', 'Jordan',
       'United Kingdom', 'Brazil', 'New Zealand', 'Canada', 'Kazakhstan',
       'United Arab Emirates', 'Australia', 'Ukraine', 'Iraq', 'France',
       'Malaysia', 'Vietnam', 'Egypt', 'Netherlands', 'Afghanistan',
       'Oman', 'Italy', 'Bahamas', 'Saudi Arabia', 'Ireland', 'Aruba',
       'Sri Lanka', 'Russia', 'Bolivia', 'Azerbaijan', 'Armenia',
       'Serbia', 'Ethiopia', 'Sweden', 'Iceland', 'China', 'Angola',
       'Germany', 'Spain', 'Tonga', 'Pakistan', 'Iran', 'Argentina',
       'Japan', 'Mexico', 'Nicaragua', 'Sierra Leone', 'Czech Republic',
       'Niger', 'Romania', 'Cyprus', 'Belgium', 'Burundi', 'Bangladesh'],Field(...,description="what is the patient's country of residence")]
    used_app_before:Annotated[Literal['yes','no'],Field(...,description="Have you used the app before")]
    # make this as a computed field result :Annotated[]
    relation:Annotated[Literal['Self', 'Others'],Field(...,description="what's your relation with the patient")]

    @computed_field
    @property
    def result(self)-> float:
        mean = 8.31
        return mean