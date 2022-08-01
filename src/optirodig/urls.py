from django.urls import path
from .views import handler,werkstoffe

app_name='optirodig'

urlpatterns = [
        path('', handler.index, name='home'),
        path('handler', handler.handler, name='handler'),
        path('handler/barchart/', handler.barchart_handler, name='barchart_handler'),
        path('handler/delete/', handler.delete_handler, name='delete_handler'),
        path('werkstoffe/', werkstoffe.werkstoffe, name='werkstoffe'),
        path('werkstoffe/stackarea/', werkstoffe.werkstoffe_stackarea, name='werkstoffe_stackarea'),
]

