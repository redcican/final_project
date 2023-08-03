from django.shortcuts import render
from optirodig.models import SchrottChemi

def chemi(request):
    chemi_obj = SchrottChemi.objects.all().order_by('id')

    return render(request, 'optirodig/chemi.html', {'chemi_obj': chemi_obj})