FROM python:3.5
MAINTAINER sakuma
#RUN pip install dash-renderer dash-core-components 
RUN pip install pandas
RUN pip install matplotlib plotly
RUN pip install dash dash-renderer dash-core-components dash-html-components

COPY demo /home/demo/
#ADD ./
EXPOSE 8050
