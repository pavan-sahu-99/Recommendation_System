import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

cus = pd.read_csv(r"bank_product_recom.csv")
feature = cus.pivot_table(index = 'Cust_Products',columns = 'Customer Id',values = 'cust_rating').fillna(0)
mat = csr_matrix(feature.values)
model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model.fit(mat)

def recom(distances, indices, index):
    prod = None
    reco = []
    for i in range(0,len(distances.flatten())):
        if i == 0:
            prod = feature.index[index].values[0]
        else:
            reco.append(feature.index[indices.flatten()[i]])
    return prod,reco

def home():
    st.session_state.butt

def main():
    st.title("Bank Product Recommendation System")
    st.write("Welcome To the Bank!!")
    id = st.text_input("Enter Your Email ID: ")
    st.write("Ex: customer15889@gmail.com")
    if st.button("Enter"):
        if id in list(cus['Customer mail'].unique()):
            st.write(f"HI  {id[:13]}!")
            index = cus[cus["Customer mail"]== id].index.values
            distances, indices = model.kneighbors(cus.iloc[index,:].values.reshape(1,-1),n_neighbors = 6)
            prod,reco = recom(distances, indices, index)
            print(f"{prod},{reco}")
            st.write(f"Hi {id[:13]}, you are using this product currently: '{prod}'")
            st.write("Please consider adding the below mentioned products too!!")
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1:
                st.write(f"1. {reco[0]}!!")
            with c2:
                st.write(f"2. {reco[1]}!!")
            with c3:
                st.write(f"3. {reco[2]}!!")
            with c4:
                st.write(f"4. {reco[3]}!!")
            with c5:
                st.write(f"5. {reco[4]}!!")
            st.button("Back")
            
        else:
            st.error("Enter a valid Email ID!!")
            st.write("Consult your nearset branch for registrations!!!")
    st.write("Thank you, Please visit again!!")
    
if __name__ == "__main__":
    main()
