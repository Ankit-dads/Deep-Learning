from django.contrib import admin
from .models import PredictionHistory


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'label', 'confidence', 'spam_prob', 'ham_prob', 'created_at', 'ip_address')
    list_filter = ('label', 'created_at')
    search_fields = ('input_text',)
    readonly_fields = ('created_at', 'cleaned_text', 'top_keywords', 'spam_prob', 'ham_prob')
    ordering = ('-created_at',)
    list_per_page = 25

    def get_queryset(self, request):
        return super().get_queryset(request)
