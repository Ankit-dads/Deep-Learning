import csv
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.db.models import Count, Avg
from django.utils import timezone
from datetime import timedelta

from .ml.predictor import predict, predict_bulk
from .models import PredictionHistory


def get_client_ip(request):
    x_forwarded = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded:
        return x_forwarded.split(',')[0]
    return request.META.get('REMOTE_ADDR')


# ─── Home / Detect Page ───────────────────────────────────────────────────────
def index(request):
    """Home page with text input form."""
    recent = PredictionHistory.objects.all()[:5]
    total = PredictionHistory.objects.count()
    spam_count = PredictionHistory.objects.filter(label='spam').count()
    ham_count = PredictionHistory.objects.filter(label='ham').count()
    context = {
        'recent': recent,
        'total': total,
        'spam_count': spam_count,
        'ham_count': ham_count,
    }
    return render(request, 'detector/index.html', context)


# ─── Single Prediction ────────────────────────────────────────────────────────
def predict_view(request):
    """Handle single-email prediction form POST."""
    if request.method == 'POST':
        text = request.POST.get('email_text', '').strip()
        if not text:
            return render(request, 'detector/index.html', {'error': 'Please enter some text.'})

        result = predict(text)

        # Save to history
        history = PredictionHistory.objects.create(
            input_text=text,
            cleaned_text=result['cleaned_text'],
            label=result['label'],
            confidence=result['confidence'],
            spam_prob=result['spam_prob'],
            ham_prob=result['ham_prob'],
            top_keywords=result['top_keywords'],
            ip_address=get_client_ip(request),
        )

        context = {
            'result': result,
            'text': text,
            'history_id': history.id,
        }
        return render(request, 'detector/result.html', context)

    return redirect('index')


# ─── Bulk Prediction ─────────────────────────────────────────────────────────
def bulk_predict_view(request):
    """Handle bulk email predictions (newline separated)."""
    if request.method == 'POST':
        raw = request.POST.get('emails_bulk', '').strip()
        if not raw:
            return render(request, 'detector/index.html', {'error': 'Please enter at least one email.'})

        lines = [l.strip() for l in raw.split('\n') if l.strip()]
        results = predict_bulk(lines)

        # Save all to history
        for text, result in zip(lines, results):
            PredictionHistory.objects.create(
                input_text=text,
                cleaned_text=result['cleaned_text'],
                label=result['label'],
                confidence=result['confidence'],
                spam_prob=result['spam_prob'],
                ham_prob=result['ham_prob'],
                top_keywords=result['top_keywords'],
                ip_address=get_client_ip(request),
            )

        paired = list(zip(lines, results))
        spam_count = sum(1 for _, r in paired if r['label'] == 'spam')
        ham_count = len(paired) - spam_count

        context = {
            'paired': paired,
            'spam_count': spam_count,
            'ham_count': ham_count,
            'total': len(paired),
        }
        return render(request, 'detector/bulk_result.html', context)

    return redirect('index')


# ─── History Page ─────────────────────────────────────────────────────────────
def history_view(request):
    """View prediction history with optional label filter."""
    label_filter = request.GET.get('label', '')
    search_query = request.GET.get('q', '')

    qs = PredictionHistory.objects.all()
    if label_filter in ('spam', 'ham'):
        qs = qs.filter(label=label_filter)
    if search_query:
        qs = qs.filter(input_text__icontains=search_query)

    total = PredictionHistory.objects.count()
    spam_count = PredictionHistory.objects.filter(label='spam').count()
    ham_count = PredictionHistory.objects.filter(label='ham').count()

    context = {
        'predictions': qs[:100],
        'total': total,
        'spam_count': spam_count,
        'ham_count': ham_count,
        'label_filter': label_filter,
        'search_query': search_query,
    }
    return render(request, 'detector/history.html', context)


# ─── Stats Page ───────────────────────────────────────────────────────────────
def stats_view(request):
    """Statistics and chart data page."""
    total = PredictionHistory.objects.count()
    spam_count = PredictionHistory.objects.filter(label='spam').count()
    ham_count = PredictionHistory.objects.filter(label='ham').count()
    avg_confidence = PredictionHistory.objects.aggregate(a=Avg('confidence'))['a'] or 0
    avg_spam_conf = PredictionHistory.objects.filter(label='spam').aggregate(a=Avg('confidence'))['a'] or 0
    avg_ham_conf = PredictionHistory.objects.filter(label='ham').aggregate(a=Avg('confidence'))['a'] or 0

    # Last 7 days daily counts
    seven_days = []
    for i in range(6, -1, -1):
        day = timezone.now().date() - timedelta(days=i)
        count = PredictionHistory.objects.filter(created_at__date=day).count()
        spam = PredictionHistory.objects.filter(created_at__date=day, label='spam').count()
        seven_days.append({
            'date': day.strftime('%b %d'),
            'total': count,
            'spam': spam,
            'ham': count - spam,
        })

    context = {
        'total': total,
        'spam_count': spam_count,
        'ham_count': ham_count,
        'avg_confidence': round(avg_confidence, 1),
        'avg_spam_conf': round(avg_spam_conf, 1),
        'avg_ham_conf': round(avg_ham_conf, 1),
        'seven_days_json': json.dumps(seven_days),
        'spam_pct': round((spam_count / total * 100), 1) if total else 0,
        'ham_pct': round((ham_count / total * 100), 1) if total else 0,
    }
    return render(request, 'detector/stats.html', context)


# ─── Export CSV ───────────────────────────────────────────────────────────────
def export_csv(request):
    """Download prediction history as CSV."""
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="spam_predictions.csv"'

    writer = csv.writer(response)
    writer.writerow(['ID', 'Label', 'Confidence (%)', 'Spam Prob (%)', 'Ham Prob (%)', 'Top Keywords', 'Input Text', 'Created At'])

    for p in PredictionHistory.objects.all():
        writer.writerow([
            p.id,
            p.label,
            p.confidence,
            p.spam_prob,
            p.ham_prob,
            ', '.join(p.top_keywords),
            p.input_text,
            p.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        ])

    return response


# ─── REST API Endpoint ────────────────────────────────────────────────────────
@csrf_exempt
@require_http_methods(["POST"])
def api_predict(request):
    """
    REST API endpoint.
    POST /api/predict/
    Body: JSON { "text": "..." }
    Returns: JSON result
    """
    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        if not text:
            return JsonResponse({'error': 'text field is required'}, status=400)

        result = predict(text)

        # Optionally save to history
        PredictionHistory.objects.create(
            input_text=text,
            cleaned_text=result['cleaned_text'],
            label=result['label'],
            confidence=result['confidence'],
            spam_prob=result['spam_prob'],
            ham_prob=result['ham_prob'],
            top_keywords=result['top_keywords'],
            ip_address=get_client_ip(request),
        )

        return JsonResponse({
            'label': result['label'],
            'confidence': result['confidence'],
            'spam_probability': result['spam_prob'],
            'ham_probability': result['ham_prob'],
            'top_keywords': result['top_keywords'],
        })
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON body'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ─── Delete History Entry ─────────────────────────────────────────────────────
def delete_prediction(request, pk):
    """Delete a single prediction from history."""
    if request.method == 'POST':
        PredictionHistory.objects.filter(pk=pk).delete()
    return redirect('history')


# ─── Clear All History ────────────────────────────────────────────────────────
def clear_history(request):
    """Clear all prediction history."""
    if request.method == 'POST':
        PredictionHistory.objects.all().delete()
    return redirect('history')
