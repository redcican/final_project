from django.db.models import Sum
from django.http import JsonResponse
from django.shortcuts import render

from optirodig.forms.handler import HandlerModelForm
from optirodig.models import *
import pandas as pd


def index(request):
    return render(request, 'optirodig/index.html')

def handler(request):
    if request.method == 'GET':
        form = HandlerModelForm()

        handler_obj = Handler.objects.all().order_by('id')

        return render(request, 'optirodig/handler.html', {'form': form, 'handler_obj': handler_obj})

    else:
        """post: create or edit a handler information"""
        fid = request.POST.get('fid', '')
        edit_object = None
        if fid.isdecimal():
            edit_object = Handler.objects.filter(id=fid).first()

        if edit_object:
            form = HandlerModelForm(data=request.POST, instance=edit_object)
        else:
            form = HandlerModelForm(data=request.POST)

        if form.is_valid():
            form.save()
            return JsonResponse({'status': True})

        return JsonResponse({'status': False, 'errors': form.errors})


def barchart_handler(request):
    """ get the total amount of schrotte and 'legierungen' for every handler """

    # calculate the total amount of schrotte for every handler
    model_list = [Schnellstahlschrott_kobaltfrei, Tiefzieh_stanzabfaelle, Schnellstahlschrott_kobaltlegiert,
                      Cr_17_Or_Cr_13, Cr_Ni_Schrott, Kaltarbeitsstahl, Warmarbeitsstahl, Cr_Ni_148xx]

    masse_sum_list = []
    firma_name_list = []

    for model in model_list:
        # calculate the total amount of schrotte for every handler
        menge_sum = model.objects.values_list('spezifikation__handler__name').annotate(menge_masse_sum=Sum('Menge'))
        firma_name_list = [item[0] for item in menge_sum]
        masse_list = [float(item[1]) for item in menge_sum]
        masse_sum_list.append(masse_list)

    df = pd.DataFrame(masse_sum_list, columns=firma_name_list)
    df.loc['Total', :] = df.sum(axis=0)
    total_sum = df.loc['Total', :].values.tolist()

    if len(total_sum) > 0:
        context = {
            "status": True,
            "data": {
                "firma_name_list": firma_name_list,
                "total_sum": total_sum,
            }
        }

        return JsonResponse(context)
    else:
        return JsonResponse({'status': False, 'errors': 'Keine Daten vorhanden'})


def delete_handler(request):
    fid = request.GET.get('fid', '')
    Handler.objects.filter(id=fid).delete()
    return JsonResponse({'status': True})
