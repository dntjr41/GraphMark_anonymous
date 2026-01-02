import os
import json
import argparse
from typing import Dict, List, Optional
import tiktoken


class TokenCountAnalyzer:
    """
    Watermark 삽입 전후의 토큰 수를 계산하고 분석하는 클래스
    """
    
    def __init__(self, encoding_name: str = 'cl100k_base'):
        """
        Initialize token count analyzer
        
        Args:
            encoding_name: tiktoken encoding name (default: 'cl100k_base' for GPT-4)
        """
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            print(f"Warning: Failed to load encoding '{encoding_name}': {e}")
            print("Falling back to 'cl100k_base'")
            self.encoding = tiktoken.get_encoding('cl100k_base')
    
    def count_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수를 계산
        
        Args:
            text: 입력 텍스트
            
        Returns:
            토큰 수
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def analyze_sample(self, data: Dict, include_paraphrased: bool = False, is_postmark: bool = False) -> Dict:
        """
        단일 샘플의 토큰 수를 분석
        
        Args:
            data: JSONL 샘플 데이터
            include_paraphrased: paraphrased_watermarked도 분석할지 여부
            is_postmark: PostMark 형식인지 여부 (text1, text2, text3 사용)
            
        Returns:
            분석 결과 딕셔너리
        """
        # PostMark 형식 처리
        if is_postmark:
            original_text = data.get('text1', '')
            watermarked_text = data.get('text2', '')
            paraphrased_text = data.get('text3', '') if include_paraphrased else ''
        else:
            # Support multiple field names for compatibility
            # Check if prompt field exists (for other_old_outputs format)
            prompt = data.get('prompt', '')
            
            # Original text: original_text or (prompt + natural_text)
            if data.get('original_text'):
                original_text = data.get('original_text', '')
            elif data.get('natural_text'):
                # For other_old_outputs: prompt + natural_text
                original_text = (prompt + ' ' + data.get('natural_text', '')).strip() if prompt else data.get('natural_text', '')
            else:
                original_text = ''
            
            # Watermarked text: prompt + watermarked_text (if prompt exists) or just watermarked_text
            watermarked_text_raw = data.get('watermarked_text', '')
            if prompt and watermarked_text_raw:
                # For other_old_outputs: prompt + watermarked_text
                watermarked_text = (prompt + ' ' + watermarked_text_raw).strip()
            else:
                watermarked_text = watermarked_text_raw
            
            # Support multiple field names for paraphrased text
            paraphrased_text = (
                data.get('paraphrased_watermarked', '') or 
                data.get('paraphrased_watermarked_text', '') or 
                data.get('paraphrased_text', '')
            )
        
        original_tokens = self.count_tokens(original_text)
        watermarked_tokens = self.count_tokens(watermarked_text)
        
        result = {
            'original_tokens': original_tokens,
            'watermarked_tokens': watermarked_tokens,
            'token_difference': watermarked_tokens - original_tokens,
            'token_ratio': watermarked_tokens / original_tokens if original_tokens > 0 else 0.0,
            'token_increase_percent': ((watermarked_tokens - original_tokens) / original_tokens * 100) if original_tokens > 0 else 0.0
        }
        
        if include_paraphrased and paraphrased_text:
            paraphrased_tokens = self.count_tokens(paraphrased_text)
            result['paraphrased_tokens'] = paraphrased_tokens
            result['paraphrased_token_difference'] = paraphrased_tokens - original_tokens
            result['paraphrased_token_ratio'] = paraphrased_tokens / original_tokens if original_tokens > 0 else 0.0
            result['paraphrased_token_increase_percent'] = ((paraphrased_tokens - original_tokens) / original_tokens * 100) if original_tokens > 0 else 0.0
            result['paraphrased_vs_watermarked_difference'] = paraphrased_tokens - watermarked_tokens
        
        return result
    
    def analyze_jsonl(self, jsonl_file: str, 
                     include_paraphrased: bool = False,
                     max_samples: Optional[int] = None) -> Dict:
        """
        JSONL 파일의 모든 샘플에 대해 토큰 수 분석
        
        Args:
            jsonl_file: 입력 JSONL 파일 경로
            include_paraphrased: paraphrased_watermarked도 분석할지 여부
            max_samples: 최대 분석 샘플 수 (None이면 전체)
            
        Returns:
            전체 통계 딕셔너리
        """
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"File not found: {jsonl_file}")
        
        # Check if this is a PostMark format file
        is_postmark = 'PostMark' in os.path.basename(jsonl_file)
        
        all_results = []
        total_samples = 0
        valid_samples = 0
        
        print(f"Analyzing token counts from: {jsonl_file}")
        print(f"Format: {'PostMark (text1/text2/text3)' if is_postmark else 'Standard'}")
        print(f"Include paraphrased: {include_paraphrased}")
        print("=" * 80)
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and total_samples >= max_samples:
                    break
                
                total_samples += 1
                
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                    continue
                
                # PostMark 형식 처리
                if is_postmark:
                    original_text = data.get('text1', '')
                    watermarked_text = data.get('text2', '')
                else:
                    # Check if required fields exist (support multiple field names)
                    # Check if prompt field exists (for other_old_outputs format)
                    prompt = data.get('prompt', '')
                    
                    # Original text: original_text or (prompt + natural_text)
                    if data.get('original_text'):
                        original_text = data.get('original_text', '')
                    elif data.get('natural_text'):
                        original_text = (prompt + ' ' + data.get('natural_text', '')).strip() if prompt else data.get('natural_text', '')
                    else:
                        original_text = ''
                    
                    # Watermarked text: prompt + watermarked_text (if prompt exists) or just watermarked_text
                    watermarked_text_raw = data.get('watermarked_text', '')
                    if prompt and watermarked_text_raw:
                        watermarked_text = (prompt + ' ' + watermarked_text_raw).strip()
                    else:
                        watermarked_text = watermarked_text_raw
                
                if not original_text or not watermarked_text:
                    continue
                
                result = self.analyze_sample(data, include_paraphrased, is_postmark=is_postmark)
                result['sample_id'] = line_num
                all_results.append(result)
                valid_samples += 1
                
                # Print progress every 50 samples
                if valid_samples % 50 == 0:
                    print(f"Processed {valid_samples} samples...")
        
        if valid_samples == 0:
            print("Error: No valid samples found!")
            return {}
        
        # Calculate aggregate statistics
        original_tokens_list = [r['original_tokens'] for r in all_results]
        watermarked_tokens_list = [r['watermarked_tokens'] for r in all_results]
        token_diff_list = [r['token_difference'] for r in all_results]
        token_ratio_list = [r['token_ratio'] for r in all_results]
        token_increase_percent_list = [r['token_increase_percent'] for r in all_results]
        
        stats = {
            'total_samples': valid_samples,
            'original_tokens': {
                'total': sum(original_tokens_list),
                'mean': sum(original_tokens_list) / len(original_tokens_list),
                'min': min(original_tokens_list),
                'max': max(original_tokens_list),
                'median': sorted(original_tokens_list)[len(original_tokens_list) // 2]
            },
            'watermarked_tokens': {
                'total': sum(watermarked_tokens_list),
                'mean': sum(watermarked_tokens_list) / len(watermarked_tokens_list),
                'min': min(watermarked_tokens_list),
                'max': max(watermarked_tokens_list),
                'median': sorted(watermarked_tokens_list)[len(watermarked_tokens_list) // 2]
            },
            'token_difference': {
                'total': sum(token_diff_list),
                'mean': sum(token_diff_list) / len(token_diff_list),
                'min': min(token_diff_list),
                'max': max(token_diff_list),
                'median': sorted(token_diff_list)[len(token_diff_list) // 2]
            },
            'token_ratio': {
                'mean': sum(token_ratio_list) / len(token_ratio_list),
                'min': min(token_ratio_list),
                'max': max(token_ratio_list),
                'median': sorted(token_ratio_list)[len(token_ratio_list) // 2]
            },
            'token_increase_percent': {
                'mean': sum(token_increase_percent_list) / len(token_increase_percent_list),
                'min': min(token_increase_percent_list),
                'max': max(token_increase_percent_list),
                'median': sorted(token_increase_percent_list)[len(token_increase_percent_list) // 2]
            }
        }
        
        if include_paraphrased:
            paraphrased_results = [r for r in all_results if 'paraphrased_tokens' in r]
            if paraphrased_results:
                paraphrased_tokens_list = [r['paraphrased_tokens'] for r in paraphrased_results]
                paraphrased_diff_list = [r['paraphrased_token_difference'] for r in paraphrased_results]
                paraphrased_ratio_list = [r['paraphrased_token_ratio'] for r in paraphrased_results]
                paraphrased_increase_percent_list = [r['paraphrased_token_increase_percent'] for r in paraphrased_results]
                paraphrased_vs_watermarked_diff_list = [r['paraphrased_vs_watermarked_difference'] for r in paraphrased_results]
                
                stats['paraphrased_tokens'] = {
                    'total': sum(paraphrased_tokens_list),
                    'mean': sum(paraphrased_tokens_list) / len(paraphrased_tokens_list),
                    'min': min(paraphrased_tokens_list),
                    'max': max(paraphrased_tokens_list),
                    'median': sorted(paraphrased_tokens_list)[len(paraphrased_tokens_list) // 2]
                }
                stats['paraphrased_token_difference'] = {
                    'total': sum(paraphrased_diff_list),
                    'mean': sum(paraphrased_diff_list) / len(paraphrased_diff_list),
                    'min': min(paraphrased_diff_list),
                    'max': max(paraphrased_diff_list),
                    'median': sorted(paraphrased_diff_list)[len(paraphrased_diff_list) // 2]
                }
                stats['paraphrased_token_ratio'] = {
                    'mean': sum(paraphrased_ratio_list) / len(paraphrased_ratio_list),
                    'min': min(paraphrased_ratio_list),
                    'max': max(paraphrased_ratio_list),
                    'median': sorted(paraphrased_ratio_list)[len(paraphrased_ratio_list) // 2]
                }
                stats['paraphrased_token_increase_percent'] = {
                    'mean': sum(paraphrased_increase_percent_list) / len(paraphrased_increase_percent_list),
                    'min': min(paraphrased_increase_percent_list),
                    'max': max(paraphrased_increase_percent_list),
                    'median': sorted(paraphrased_increase_percent_list)[len(paraphrased_increase_percent_list) // 2]
                }
                stats['paraphrased_vs_watermarked_difference'] = {
                    'mean': sum(paraphrased_vs_watermarked_diff_list) / len(paraphrased_vs_watermarked_diff_list),
                    'min': min(paraphrased_vs_watermarked_diff_list),
                    'max': max(paraphrased_vs_watermarked_diff_list),
                    'median': sorted(paraphrased_vs_watermarked_diff_list)[len(paraphrased_vs_watermarked_diff_list) // 2]
                }
                stats['paraphrased_samples'] = len(paraphrased_results)
        
        stats['per_sample_results'] = all_results
        
        return stats
    
    def print_statistics(self, stats: Dict):
        """
        통계 결과를 출력
        
        Args:
            stats: analyze_jsonl에서 반환된 통계 딕셔너리
        """
        if not stats:
            print("No statistics to display")
            return
        
        print("\n" + "=" * 80)
        print("Token Count Analysis Results")
        print("=" * 80)
        
        print(f"\nTotal Samples Analyzed: {stats['total_samples']}")
        
        print("\n" + "-" * 80)
        print("Original Text Tokens")
        print("-" * 80)
        orig = stats['original_tokens']
        print(f"  Total:     {orig['total']:,}")
        print(f"  Mean:      {orig['mean']:.2f}")
        print(f"  Median:    {orig['median']:.2f}")
        print(f"  Min:       {orig['min']:,}")
        print(f"  Max:       {orig['max']:,}")
        
        print("\n" + "-" * 80)
        print("Watermarked Text Tokens")
        print("-" * 80)
        wm = stats['watermarked_tokens']
        print(f"  Total:     {wm['total']:,}")
        print(f"  Mean:      {wm['mean']:.2f}")
        print(f"  Median:    {wm['median']:.2f}")
        print(f"  Min:       {wm['min']:,}")
        print(f"  Max:       {wm['max']:,}")
        
        print("\n" + "-" * 80)
        print("Token Difference (Watermarked - Original)")
        print("-" * 80)
        diff = stats['token_difference']
        print(f"  Total:     {diff['total']:,}")
        print(f"  Mean:      {diff['mean']:.2f}")
        print(f"  Median:    {diff['median']:.2f}")
        print(f"  Min:       {diff['min']:,}")
        print(f"  Max:       {diff['max']:,}")
        
        print("\n" + "-" * 80)
        print("Token Ratio (Watermarked / Original)")
        print("-" * 80)
        ratio = stats['token_ratio']
        print(f"  Mean:      {ratio['mean']:.4f}")
        print(f"  Median:    {ratio['median']:.4f}")
        print(f"  Min:       {ratio['min']:.4f}")
        print(f"  Max:       {ratio['max']:.4f}")
        
        print("\n" + "-" * 80)
        print("Token Increase Percentage")
        print("-" * 80)
        inc = stats['token_increase_percent']
        print(f"  Mean:      {inc['mean']:.2f}%")
        print(f"  Median:    {inc['median']:.2f}%")
        print(f"  Min:       {inc['min']:.2f}%")
        print(f"  Max:       {inc['max']:.2f}%")
        
        if 'paraphrased_tokens' in stats:
            print("\n" + "-" * 80)
            print(f"Paraphrased Text Tokens (from {stats.get('paraphrased_samples', 0)} samples)")
            print("-" * 80)
            para = stats['paraphrased_tokens']
            print(f"  Total:     {para['total']:,}")
            print(f"  Mean:      {para['mean']:.2f}")
            print(f"  Median:    {para['median']:.2f}")
            print(f"  Min:       {para['min']:,}")
            print(f"  Max:       {para['max']:,}")
            
            print("\n" + "-" * 80)
            print("Paraphrased Token Difference (Paraphrased - Original)")
            print("-" * 80)
            para_diff = stats['paraphrased_token_difference']
            print(f"  Total:     {para_diff['total']:,}")
            print(f"  Mean:      {para_diff['mean']:.2f}")
            print(f"  Median:    {para_diff['median']:.2f}")
            print(f"  Min:       {para_diff['min']:,}")
            print(f"  Max:       {para_diff['max']:,}")
            
            print("\n" + "-" * 80)
            print("Paraphrased vs Watermarked Token Difference")
            print("-" * 80)
            para_vs_wm = stats['paraphrased_vs_watermarked_difference']
            print(f"  Mean:      {para_vs_wm['mean']:.2f}")
            print(f"  Median:    {para_vs_wm['median']:.2f}")
            print(f"  Min:       {para_vs_wm['min']:,}")
            print(f"  Max:       {para_vs_wm['max']:,}")
        
        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token counts before and after watermark insertion"
    )
    parser.add_argument(
        '--jsonl_file',
        type=str,
        required=True,
        help="Path to JSONL file with watermarked examples"
    )
    parser.add_argument(
        '--include_paraphrased',
        action='store_true',
        help="Also analyze paraphrased_watermarked text"
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help="Maximum number of samples to analyze (default: all)"
    )
    parser.add_argument(
        '--encoding',
        type=str,
        default='cl100k_base',
        help="tiktoken encoding name (default: cl100k_base for GPT-4)"
    )
    parser.add_argument(
        '--save_results',
        type=str,
        default=None,
        help="Path to save detailed results as JSON file"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TokenCountAnalyzer(encoding_name=args.encoding)
    
    # Analyze JSONL file
    stats = analyzer.analyze_jsonl(
        jsonl_file=args.jsonl_file,
        include_paraphrased=args.include_paraphrased,
        max_samples=args.max_samples
    )
    
    # Print statistics
    analyzer.print_statistics(stats)
    
    # Save results if requested
    if args.save_results:
        # Remove per_sample_results for file size (can be large)
        stats_to_save = {k: v for k, v in stats.items() if k != 'per_sample_results'}
        with open(args.save_results, 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to: {args.save_results}")
        
        # Optionally save per-sample results to a separate file
        if 'per_sample_results' in stats:
            per_sample_file = args.save_results.replace('.json', '_per_sample.json')
            with open(per_sample_file, 'w', encoding='utf-8') as f:
                json.dump(stats['per_sample_results'], f, indent=2, ensure_ascii=False)
            print(f"Per-sample results saved to: {per_sample_file}")


if __name__ == "__main__":
    main()

