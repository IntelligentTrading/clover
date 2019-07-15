from django.conf.urls import url

from apps.doge.views.doge_view import CommitteesView
from apps.doge.views.allocations_history_view import AllocationsHistoryView
from apps.doge.views.doge_health_status_view import DogeHealthView

app_name = 'doge'

urlpatterns = [

    url(r'^dashboard$', CommitteesView.as_view(), name='dashboard'),
    url(r'^dashboard/(?P<ticker>\w+)/$', CommitteesView.as_view(), name='dashboard'),
    url(r'^dashboard/(?P<ticker>\w+)/(?P<hours>[0-9]+)/$', CommitteesView.as_view(), name='dashboard'),
    url(r'^allocations$', AllocationsHistoryView.as_view(), name='allocations'),
    url(r'^health/(?P<hours>[0-9]+)/$', DogeHealthView.as_view(), name='health'),
    #url(r'^exchange_account$', ExchangeAccountView.as_view(), name='exchange_account'),
    #url(r'^allocation$', AllocationsView.as_view(), name='allocation'),

]
