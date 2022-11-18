import numpy as np
import pandas as pd
import pickle
import streamlit as st


# load data
data = pickle.load(open('x_test.pkl','rb'))
data_ = pd.DataFrame(data)

# load model;
sig_clf = pickle.load(open('model.pkl','rb'))

X_test = pickle.load(open('array.pkl','rb'))

st.title('Cancer Diagnosis System')
st.header('How Genes and tumour are related:')
st.markdown('Genes control how your cells work by making proteins. The proteins have specific functions and '
            'act as messengers for the cell. Each gene must have the correct instructions for making its protein.')
st.markdown('All cancers begin when one or more genes in a cell mutate. A mutation is a change. ' \
        'It creates an abnormal protein or it may prevent a protein’s formation. An abnormal protein provides ' \
        'different information than a normal protein. This can cause cells to multiply uncontrollably and become cancerous.')

st.header('About ML model:')
st.markdown('The type of cancer is classified into 9 classes, the ML model predicts the particular class '
            'based on given input of (Gene) and (Variation). You can also check the clinical evidence present '
            'with respect to the selected Gene and Variation type. ')

gene_ = st.selectbox('Select the gene type here',
                    list(data_['Gene'].unique()))
gene = str(gene_)

variation_ = st.selectbox('Select the variation type here',
                    list(data_['Variation'].unique()))
variation = str(variation_)
x1 = data_.loc[(data_['Gene'] == gene) & (data_['Variation'] == variation)]

def predict(gene,variation):
    if x1.empty:
        c1 = data_.groupby('Gene')
        c2 = c1.get_group(str(gene))
        st.warning('Please select some other combination')
        st.write('The following variations might be useful to choose from:')
        st.table(c2['Variation'].values)
    else:
        f1 = X_test.toarray()[x1.index[0]]
        predict = list(sig_clf.predict([f1]))[0]
        #print('This Gene & Variation belong to: Class {}'.format(predict))
        return ('This Gene & Variation belong to: Class {}'.format(predict))

if st.button('Predict Class'):
    predict_cls = predict(gene,variation)
    st.success(predict_cls)
if st.button("Clinical Evidence"):
    txt = st.text_area('Detailed evidence:',data_['TEXT'].values[x1.index[0]],height = 30)
