from django.db import models


class PredictionHistory(models.Model):
    LABEL_CHOICES = [('spam', 'Spam'), ('ham', 'Ham')]

    input_text = models.TextField(verbose_name="Input Text")
    cleaned_text = models.TextField(verbose_name="Cleaned Text", blank=True)
    label = models.CharField(max_length=4, choices=LABEL_CHOICES)
    confidence = models.FloatField(help_text="Confidence % (0–100)")
    spam_prob = models.FloatField(default=0.0)
    ham_prob = models.FloatField(default=0.0)
    top_keywords = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Prediction'
        verbose_name_plural = 'Prediction History'

    def __str__(self):
        preview = self.input_text[:60] + '...' if len(self.input_text) > 60 else self.input_text
        return f"[{self.label.upper()}] {preview}"

    @property
    def is_spam(self):
        return self.label == 'spam'
