from django.shortcuts import render
from optirodig.models import Steel

def stahl(request):
    steel_obj = Steel.objects.all().order_by('id')

    return render(request, 'optirodig/stahl.html', {'steel_obj': steel_obj})
