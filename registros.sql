-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Tempo de geração: 24/10/2025 às 18:26
-- Versão do servidor: 10.4.32-MariaDB
-- Versão do PHP: 8.0.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Banco de dados: `reconhecimento_facial`
--

-- --------------------------------------------------------

--
-- Estrutura para tabela `registros`
--

CREATE TABLE `registros` (
  `id` int(11) NOT NULL,
  `nome` varchar(80) DEFAULT NULL,
  `confianca` int(11) DEFAULT NULL,
  `data_hora` datetime DEFAULT NULL,
  `caminho_imagem` varchar(300) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Despejando dados para a tabela `registros`
--

INSERT INTO `registros` (`id`, `nome`, `confianca`, `data_hora`, `caminho_imagem`) VALUES
(267, 'Arthur Oliveira Leal', 66, '2025-10-24 11:26:18', 'rostos_salvos\\arthur oliveira leal\\2025-10-24_11-26-18_arthur oliveira leal_143_140.png'),
(269, 'Yuri Christian', 70, '2025-10-24 11:32:46', 'rostos_salvos\\yuri christian\\2025-10-24_11-32-46_yuri christian_191_159.png'),
(271, 'Daniel Abner', 66, '2025-10-24 11:53:53', 'rostos_salvos\\daniel abner\\2025-10-24_11-53-53_daniel abner_291_169.png'),
(272, 'Emanuel Rezende', 69, '2025-10-24 13:07:05', 'rostos_salvos\\emanuel rezende\\2025-10-24_13-07-05_emanuel rezende_321_148.png'),
(281, 'Antonio Bezerra', 78, '2025-10-24 13:24:20', 'rostos_salvos\\antonio bezerra\\2025-10-24_13-24-20_antonio bezerra_146_164.png');

--
-- Índices para tabelas despejadas
--

--
-- Índices de tabela `registros`
--
ALTER TABLE `registros`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT para tabelas despejadas
--

--
-- AUTO_INCREMENT de tabela `registros`
--
ALTER TABLE `registros`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=282;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
