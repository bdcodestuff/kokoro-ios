enum G2PProcessorError : Error {
  case processorNotInitialized
  case unsupportedLanguage
}

protocol G2PProcessor {
  func setLanguage(_ language: Language) throws
  func process(input: String) throws -> String

  #if canImport(MisakiSwift)
  func processWithTokens(input: String) throws -> (phonemes: String, tokens: [MisakiSwift.MToken])
  #endif
}
