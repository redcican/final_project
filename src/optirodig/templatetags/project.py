from django.template import Library
from django.shortcuts import reverse

register = Library()


@register.inclusion_tag('inclusion/manage_menu_list.html')
def manage_menu_list(request):
    data_list = [
        {'title': 'Händler', 'url': reverse('optirodig:handler')},
        {'title': 'Werkstoffe', 'url': reverse('optirodig:werkstoffe')},
        {'title': 'Schrott', 'url': reverse('optirodig:schrott')},
        {'title': 'Stahl', 'url': reverse('optirodig:stahl')},
        {'title': 'Chemi', 'url': reverse('optirodig:chemi')},
        {'title': 'Optimierungstool', 'url': reverse('main:ml_home')},
        # {'title': 'Optimierungstool', 'url': reverse('main:optimierung')},
        # {'title': 'File', 'url': reverse('tracer:file', kwargs={'project_id': request.tracer.project.id})},
        # {'title': 'Settings', 'url': reverse('tracer:setting', kwargs={'project_id': request.tracer.project.id})},
    ]

    for item in data_list:
        if request.path_info.startswith(item['url']):
            item['class'] = 'active'

    return {'data_list': data_list}