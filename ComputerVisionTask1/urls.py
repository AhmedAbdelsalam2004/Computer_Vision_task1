from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path("api/state/",          views.StateView.as_view(),         name="api-state"),
    path("api/upload/",         views.UploadView.as_view(),        name="api-upload"),
    path("api/apply-filter/",   views.ApplyFilterView.as_view(),   name="api-apply-filter"),
    path("api/undo/",           views.UndoView.as_view(),          name="api-undo"),
    path("api/reset/",          views.ResetView.as_view(),         name="api-reset"),
    path("api/histogram/",      views.DrawHistogramView.as_view(), name="api-histogram"),
    path("api/equalize/",       views.EqualizeView.as_view(),      name="api-equalize"),
    path("api/switch-mode/",    views.SwitchModeView.as_view(),    name="api-switch-mode"),
    path("api/hybrid-low/",     views.HybridLowUploadView.as_view(),  name="api-hybrid-low"),
    path("api/hybrid-high/",    views.HybridHighUploadView.as_view(), name="api-hybrid-high"),
    path("api/hybrid-mix/",     views.HybridMixView.as_view(),        name="api-hybrid-mix"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)