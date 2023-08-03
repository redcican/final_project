from django.shortcuts import render, redirect
from optirodig.models import Schrott, GiessereiSchrott
from optirodig.forms.schrott import SchrottModelForm
from django.http import JsonResponse
import numpy as np
from django.db.models import F


def schrott(request):
    """
    Schrott View
    """
    if request.POST.get("price_simulate"):
        obj = Schrott.objects.defer('price')
        for p in obj:
            upper = float(p.price) * 1.2
            lower = float(p.price) * 0.8
            p.price = np.random.uniform(lower, upper)
            p.save()
        return redirect("optirodig:schrott")
    
    if request.method == 'GET':
        form = SchrottModelForm()
        schrott_obj = Schrott.objects.all().order_by('id')
        giesserei_schrott_obj = GiessereiSchrott.objects.all().order_by('id')
        return render(request, 'optirodig/schrott.html', {'form':form, 'schrott_obj': schrott_obj, 'giesserei_schrott_obj': giesserei_schrott_obj})

    else:
        """post: create or edit a schrott information""" 

        fid = request.POST.get('fid', '')
        edit_object = None
        if fid.isdecimal():
            edit_object = Schrott.objects.filter(id=fid).first()
        if edit_object:
            form = SchrottModelForm(data=request.POST, instance=edit_object)
        else:
            form = SchrottModelForm(data=request.POST)
            
        if form.is_valid():
            form.save()
            return JsonResponse({'status': True})
        
        return JsonResponse({'status': False, 'errors': form.errors})