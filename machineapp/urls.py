from django.urls import path
from machineapp import views

urlpatterns=[
    path("",views.home),
    path("file/",views.file_upload, name="file_upload"),
    path("result/",views.predict,name="predict")


]