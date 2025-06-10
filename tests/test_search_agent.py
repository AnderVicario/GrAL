import unittest
from unittest.mock import patch, MagicMock, ANY
import json
from datetime import datetime

from agents.search_agent import SearchAgent, _configure_logging


class TestSearchAgent(unittest.TestCase):

    def setUp(self):
        """Proba-ingurunea ezarri test metodo bakoitzaren aurretik."""
        # Simulatu dependentziak eta sortu test instantzia
        self.patcher1 = patch('agents.search_agent.Together')
        self.mock_together = self.patcher1.start()

        # Simulatu erantzuna chat completions-entzat
        self.mock_response = MagicMock()
        self.mock_response.choices = [MagicMock()]
        self.mock_response.choices[0].delta.content = "test content"

        # Konfiguratu Together bezero simulatuaren erantzunak
        self.mock_together.return_value.chat.completions.create.return_value = [self.mock_response]

        # Ezarri test instantzia
        self.agent = SearchAgent("Test prompt", "en")

        # Ordezkatu benetako Together bezeroa simulatuarekin
        self.agent.llm_client = self.mock_together.return_value

    def tearDown(self):
        """Garbitu test metodo bakoitzaren ondoren."""
        self.patcher1.stop()

    def test_init(self):
        """Testeatu SearchAgent-en hasieratzea."""
        self.assertEqual(self.agent.user_prompt, "Test prompt")
        self.assertEqual(self.agent.user_prompt_language, "en")
        self.assertEqual(self.agent.model_name, "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free")
        self.assertEqual(self.agent.entities, [])
        self.assertIsNone(self.agent.horizon)
        self.assertIsNone(self.agent.start_date)
        self.assertIsInstance(self.agent.end_date, datetime)
        self.assertIsNone(self.agent.expiration_date)
        self.assertIsNone(self.agent.date_range)

    @patch('agents.search_agent.re.sub')
    def test_identify_entities(self, mock_re_sub):
        """Testeatu _identify_entities metodoa."""
        # Konfiguratu simulazioa
        mock_re_sub.return_value = '[{"name": "Test Company", "ticker": "TEST"}]'

        # Exekutatu metodoa
        result = self.agent._identify_entities()

        # Emaitzak egiaztatu
        self.assertEqual(result, '[{"name": "Test Company", "ticker": "TEST"}]')
        self.mock_together.return_value.chat.completions.create.assert_called_once()
        mock_re_sub.assert_called_once()

    @patch('agents.search_agent.re.sub')
    @patch('agents.search_agent.json.JSONDecoder')
    def test_set_dates_success(self, mock_json_decoder, mock_re_sub):
        """Testeatu _set_dates metodoa JSON parseaketa arrakastatsuarekin."""
        # Konfiguratu simulazioak
        mock_re_sub.return_value = '{"horizon": "medium", "start_date": "2023-01-01", "end_date": "2023-04-01", "expiration_date": "2023-05-01"}'
        mock_decoder = MagicMock()
        mock_decoder.raw_decode.return_value = (
            {"horizon": "medium", "start_date": "2023-01-01", "end_date": "2023-04-01",
             "expiration_date": "2023-05-01"},
            100
        )
        mock_json_decoder.return_value = mock_decoder

        # Exekutatu metodoa
        self.agent._set_dates()

        # Emaitzak egiaztatu
        self.assertEqual(self.agent.horizon, "medium")
        self.assertEqual(self.agent.start_date, "2023-01-01")
        self.assertEqual(self.agent.end_date, "2023-04-01")
        self.assertEqual(self.agent.expiration_date, "2023-05-01")

    @patch('agents.search_agent.re.sub')
    @patch('agents.search_agent.json.JSONDecoder')
    @patch('agents.search_agent.logging.error')
    def test_set_dates_error(self, mock_logging, mock_json_decoder, mock_re_sub):
        """Testeatu _set_dates metodoa JSON parse errorearekin."""
        # Konfiguratu simulazioak
        mock_re_sub.return_value = 'invalid json'
        mock_decoder = MagicMock()
        mock_decoder.raw_decode.side_effect = ValueError("Invalid JSON")
        mock_json_decoder.return_value = mock_decoder

        # Exekutatu metodoa
        self.agent._set_dates()

        # Erroreak tratatzea egiaztatu
        mock_logging.assert_called_once()
        self.assertEqual(self.agent.horizon, "medium")  # Lehentasunezko balioa

    @patch('agents.search_agent.re.sub')
    def test_distil_query(self, mock_re_sub):
        """Testeatu _distil_query metodoa."""
        # Konfiguratu simulazioa
        mock_re_sub.return_value = "Refined query"

        # Sortu entitate simulatua
        mock_entity = MagicMock()
        mock_entity.name = "Test Entity"

        # Exekutatu metodoa
        result = self.agent._distil_query(mock_entity)

        # Emaitzak egiaztatu
        self.assertEqual(result, "Refined query")
        self.mock_together.return_value.chat.completions.create.assert_called_once()

    @patch('agents.search_agent.re.search')
    @patch('agents.search_agent.json.loads')
    @patch('agents.search_agent.FinancialEntity')
    @patch('agents.search_agent.logging.info')
    def test_process_entities_success(self, mock_logging, mock_financial_entity, mock_json_loads, mock_re_search):
        """Testeatu _process_entities metodoa JSON parseaketa arrakastatsuarekin."""
        # Konfiguratu simulazioak
        mock_match = MagicMock()
        mock_match.group.return_value = '[{"name": "Test Entity", "ticker": "TEST"}]'
        mock_re_search.return_value = mock_match
        mock_json_loads.return_value = [{"name": "Test Entity", "ticker": "TEST"}]

        # Exekutatu metodoa _set_dates eta _identify_entities patchatuta
        with patch.object(self.agent, '_set_dates') as mock_set_dates, \
             patch.object(self.agent, '_identify_entities',
                          return_value='[{"name": "Test Entity", "ticker": "TEST"}]') as mock_identify:
            self.agent._process_entities()

        # Emaitzak egiaztatu
        mock_set_dates.assert_called_once()
        mock_identify.assert_called_once()
        mock_financial_entity.assert_called_once()
        self.assertEqual(len(self.agent.entities), 1)

    @patch('agents.search_agent.re.search')
    @patch('agents.search_agent.logging.error')
    def test_process_entities_no_json(self, mock_logging, mock_re_search):
        """Testeatu _process_entities metodoa erantzunean JSONik aurkitu ez denean."""
        # Konfiguratu simulazioak
        mock_re_search.return_value = None

        # Exekutatu metodoa _set_dates eta _identify_entities patchatuta
        with patch.object(self.agent, '_set_dates') as mock_set_dates, \
             patch.object(self.agent, '_identify_entities', return_value='No JSON here') as mock_identify:
            self.agent._process_entities()

        # Erroreak tratatzea egiaztatu
        mock_logging.assert_called_once_with("No valid JSON structure was found in the response.")
        self.assertEqual(self.agent.entities, [])

    @patch('agents.search_agent.re.search')
    @patch('agents.search_agent.json.loads')
    @patch('agents.search_agent.logging.error')
    def test_process_entities_json_error(self, mock_logging, mock_json_loads, mock_re_search):
        """Testeatu _process_entities metodoa JSON dekodifikazio errorearekin."""
        # Konfiguratu simulazioak
        mock_match = MagicMock()
        mock_match.group.return_value = 'invalid json'
        mock_re_search.return_value = mock_match
        mock_json_loads.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)

        # Exekutatu metodoa _set_dates eta _identify_entities patchatuta
        with patch.object(self.agent, '_set_dates') as mock_set_dates, \
             patch.object(self.agent, '_identify_entities', return_value='[invalid json]') as mock_identify:
            self.agent._process_entities()

        # Erroreak tratatzea egiaztatu
        mock_logging.assert_called_once()
        self.assertEqual(self.agent.entities, [])

    def test_rerank_documents_success(self):
        """Testeatu _rerank_documents metodoa berrrankina arrakastaz egiten denean."""
        # Konfiguratu simulazioa
        mock_rerank_response = MagicMock()
        mock_rerank_response.results = [MagicMock(index=1), MagicMock(index=0)]
        self.mock_together.return_value.rerank.create.return_value = mock_rerank_response

        # Exekutatu metodoa
        result = self.agent._rerank_documents("query", ["doc1", "doc2"], 2)

        # Emaitzak egiaztatu
        self.assertEqual(result, ["doc2", "doc1"])
        self.mock_together.return_value.rerank.create.assert_called_once()

    @patch('agents.search_agent.logging.error')
    def test_rerank_documents_error(self, mock_logging):
        """Testeatu _rerank_documents metodoa berrrankinean errore bat gertatzen denean."""
        # Konfiguratu simulazioa salbuespena botatzeko
        self.mock_together.return_value.rerank.create.side_effect = Exception("Reranking error")

        # Exekutatu metodoa
        result = self.agent._rerank_documents("query", ["doc1", "doc2", "doc3"], 2)

        # Lehenetsi erabili den portaera egiaztatu
        self.assertEqual(result, ["doc1", "doc2"])
        mock_logging.assert_called_once()

    def test_handle_semantic_search(self):
        """Testeatu _handle_semantic_search metodoa."""
        # Sortu entitate simulatua eta ezarri itzulitako balioak
        mock_entity = MagicMock()
        mock_entity.semantic_search.return_value = [
            {'text': 'Entity result 1', 'metadata': {'source': 'test1'}}
        ]

        # Simulatu datu-base globala
        with patch('agents.search_agent.VectorMongoDB') as mock_mongo:
            mock_db_instance = MagicMock()
            mock_db_instance.semantic_search.return_value = [
                {'text': 'Global result 1', 'metadata': {'source': 'test2'}}
            ]
            mock_mongo.return_value = mock_db_instance

            # Simulatu berrrankina
            with patch.object(self.agent, '_rerank_documents') as mock_rerank:
                mock_rerank.return_value = ['Entity result 1', 'Global result 1']

                # Exekutatu metodoa
                result = self.agent._handle_semantic_search(mock_entity, "test query")

        # Emaitzak egiaztatu
        self.assertEqual(len(result['reranked_results']), 2)
        mock_entity.semantic_search.assert_called_once()
        mock_entity.drop_vector_index.assert_called_once()

    @patch('agents.search_agent.VectorMongoDB')
    @patch('agents.search_agent.AnalysisAgent')
    @patch('agents.search_agent.MarkdownAgent')
    @patch('agents.search_agent.FundamentalAnalysisAgent')
    def test_process_all(self, mock_fundamental_agent, mock_markdown_agent, mock_analysis_agent, mock_mongodb):
        """Testeatu process_all metodoa."""
        # Sortu entitate simulatuak
        mock_entity = MagicMock()
        mock_entity.name = "Test Entity"
        mock_entity.ticker = "TEST"
        self.agent.entities = [mock_entity]

        # Simulatu FundamentalAnalysisAgent
        mock_fundamental_instance = MagicMock()
        mock_fundamental_instance.process.return_value = "Fundamental analysis"
        mock_fundamental_agent.return_value = mock_fundamental_instance

        # Simulatu semantic search tratamendua
        with patch.object(self.agent, '_process_entities') as mock_process_entities, \
             patch.object(self.agent, '_distil_query', return_value="Refined query") as mock_distil, \
             patch.object(self.agent, '_handle_semantic_search',
                          return_value={"reranked_results": []}) as mock_search:
            # Simulatu agenteak
            mock_analysis_instance = MagicMock()
            mock_analysis_instance.generate_final_analysis.return_value = "Final analysis"
            mock_analysis_agent.return_value = mock_analysis_instance

            mock_markdown_instance = MagicMock()
            mock_markdown_instance.generate_markdown.return_value = "Markdown content"
            mock_markdown_agent.return_value = mock_markdown_instance

            # Exekutatu metodoa
            result = self.agent.process_all(advanced_mode=False)

        # Emaitzak egiaztatu
        mock_process_entities.assert_called_once()
        mock_distil.assert_called_once()
        mock_search.assert_called_once()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['entity_name'], "Test Entity")
        self.assertEqual(result[0]['content'], "Markdown content")
        self.assertEqual(result[0]['ticker'], "TEST")


class TestConfigureLogging(unittest.TestCase):

    @patch('agents.search_agent.colorlog.getLogger')
    @patch('agents.search_agent.colorlog.StreamHandler')
    @patch('agents.search_agent.colorlog.ColoredFormatter')
    def test_configure_logging(self, mock_formatter, mock_handler, mock_get_logger):
        """Testeatu _configure_logging funtzioa."""
        # Konfiguratu simulazioak
        mock_logger = MagicMock()
        mock_logger.handlers = [MagicMock()]
        mock_get_logger.return_value = mock_logger

        mock_handler_instance = MagicMock()
        mock_handler.return_value = mock_handler_instance

        mock_formatter_instance = MagicMock()
        mock_formatter.return_value = mock_formatter_instance

        # Exekutatu funtzioa
        _configure_logging()

        # Emaitzak egiaztatu
        mock_get_logger.assert_called_once()
        mock_logger.setLevel.assert_called_once()
        mock_logger.removeHandler.assert_called_once()
        mock_handler.assert_called_once()
        mock_formatter.assert_called_once()
        mock_handler_instance.setFormatter.assert_called_once_with(mock_formatter_instance)
        mock_logger.addHandler.assert_called_once_with(mock_handler_instance)
        self.assertFalse(mock_logger.propagate)


if __name__ == '__main__':
    unittest.main()
