from django.conf.urls import url

from apps.doge.views.doge_view import CommitteesView
app_name = 'doge'

urlpatterns = [

    url(r'^dashboard/(?P<ticker>\w+)/(?P<hours>[0-9]+)/$', CommitteesView.as_view(), name='dashboard'),
    #url(r'^exchange_account$', ExchangeAccountView.as_view(), name='exchange_account'),
    #url(r'^allocation$', AllocationsView.as_view(), name='allocation'),

]
