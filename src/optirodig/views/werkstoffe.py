import uuid
from django.shortcuts import render, reverse, redirect
from django.http import JsonResponse
from django.db.models import Sum

from optirodig.forms.handler import UploadWerkstoffeDateiModelForm
from utils.transform import get_werkstoffe_url
from utils.update_werkstoffe_from_upload import upload_werkstoffe
from optirodig.models import *
import numpy as np
from django.core.files.storage import default_storage

models_dict = {
    'Tiefzieh_stanzabfaelle': Tiefzieh_stanzabfaelle,
    'Schnellstahlschrott_kobaltlegiert': Schnellstahlschrott_kobaltlegiert,
    'Schnellstahlschrott_kobaltfrei': Schnellstahlschrott_kobaltfrei,
    'Cr_17_Or_Cr_13': Cr_17_Or_Cr_13,
    'Cr_Ni_Schrott': Cr_Ni_Schrott,
    'Kaltarbeitsstahl': Kaltarbeitsstahl,
    'Warmarbeitsstahl': Warmarbeitsstahl,
    'Cr_Ni_148xx': Cr_Ni_148xx
}

def werkstoffe(request):
    """
    upload Werkstoffe-View
    """
    if request.method == 'GET':
        upload_form = UploadWerkstoffeDateiModelForm()

        return render(request, 'optirodig/werkstoffe.html', {'upload_form': upload_form})

    # POST-Request
    upload_form = UploadWerkstoffeDateiModelForm(data=request.POST, files=request.FILES)
    if upload_form.is_valid():
        
        upload_file_object = upload_form.cleaned_data['upload_file']
        file_id = uuid.uuid1()
        file_path = get_werkstoffe_url(file_id)
       
        # Save file into local
        default_storage.save(file_path, upload_file_object)
        
        instance = SchrottWerkstoffe.objects.create(
            handler=upload_form.cleaned_data["handler"],
            form=upload_form.cleaned_data["form"],
            spezifikation=upload_form.cleaned_data["spezifikation"],
            upload_file=file_path,
        )
        # create or update spezifische werkstoffe database
        foreign_key = instance.id
        file_path = file_path
        table_name = instance.get_spezifikation_display()
        upload_model = models_dict[table_name]
        upload_werkstoffe(foreign_key, file_path, upload_model._meta.db_table)

        url = reverse('optirodig:werkstoffe')
        return redirect(url)
    return render(request, 'optirodig/werkstoffe.html', {'upload_form': upload_form})



def werkstoffe_stackarea(request):
    """ get every spezfiikation from werkstoffe database for every firma"""
    spezfikation_menge_list = []
    for model in models_dict.values():
        spezifikation_menge_sum = model.objects.values_list('spezifikation__handler__name').annotate(
            menge_masse_sum=Sum('Menge'))
        firma_name_list = [item[0] for item in spezifikation_menge_sum]
        masse_list = [float(item[1]) for item in spezifikation_menge_sum]
        spezfikation_menge_list.append(masse_list)

    spezikation_names = list(models_dict.keys())
    menge_list = np.array(spezfikation_menge_list).T.tolist()

    if len(firma_name_list) > 0:
        context = {
            "status": True,
            "data": {
                "spezikation_names": spezikation_names,
                "firma_name_list": firma_name_list,
                "menge_list": menge_list
            }
        }
        return JsonResponse(context)
    else:
        return JsonResponse({"status": False, "errors": "Keine Daten vorhanden"})