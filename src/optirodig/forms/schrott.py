from django import forms
from optirodig.models import Schrott
from optirodig.forms.bootstrap import BootstrapForm

class SchrottModelForm(BootstrapForm, forms.ModelForm):
    """
    Schrott Model Form
    """
    class Meta:
        model = Schrott
        fields = '__all__'